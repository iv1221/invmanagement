import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import signal
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
    app_callback_class,
    GStreamerDetectionApp
)
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # { label: [confidence, confidence, ...] }
        self.confidence_history = defaultdict(list)
        self.fps_history = []
        self.last_time = time.time()
        self.excluded_labels = {
            "person"
        }

    def plot_confidence(self):
        if not self.confidence_history:
            print("No detections recorded, nothing to plot.")
            return

        # Save to the same directory as the script
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confidence_summary.png")

        labels = sorted(self.confidence_history.keys())
        avg_confidences = [np.mean(self.confidence_history[l]) for l in labels]
        overall_avg = np.mean([c for confs in self.confidence_history.values() for c in confs])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Detection Confidence Summary", fontsize=14, fontweight="bold")

        bars = axes[0].bar(labels, avg_confidences, color="steelblue", edgecolor="black")
        axes[0].axhline(overall_avg, color="red", linestyle="--", linewidth=1.5, label=f"Overall avg: {overall_avg:.2f}")
        axes[0].set_title("Average Confidence per Object")
        axes[0].set_xlabel("Object Label")
        axes[0].set_ylabel("Average Confidence")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend()

        for bar, val in zip(bars, avg_confidences):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9
            )

        ax2 = axes[1]
        ax2_fps = ax2.twinx()  # second y-axis for FPS

        for label in labels:
            ax2.plot(self.confidence_history[label], label=label, alpha=0.75)
        ax2.axhline(overall_avg, color="red", linestyle="--", linewidth=1.5, label=f"Overall avg: {overall_avg:.2f}")
        ax2.set_title("Confidence & FPS Over Time")
        ax2.set_xlabel("Detection Event")
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1.05)

        if self.fps_history:
            total_detections = max(len(v) for v in self.confidence_history.values())
            fps_x = np.linspace(0, total_detections - 1, len(self.fps_history))
            ax2_fps.plot(fps_x, self.fps_history, color="orange", alpha=0.5, 
                         linewidth=0.8, label="FPS")
            
            avg_fps = np.mean(self.fps_history)
            ax2_fps.axhline(avg_fps, color="orange", linestyle="--", linewidth=1.5,
                            label=f"Avg FPS: {avg_fps:.1f}")
            
            ax2_fps.set_ylabel("FPS", color="orange")
            ax2_fps.tick_params(axis="y", labelcolor="orange")
            ax2_fps.set_ylim(0, max(self.fps_history) * 1.2)
            ax2_fps.legend(loc="upper right")
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)  # Free memory after saving
        print(f"Graph saved to {save_path}")


def app_callback(pad, buffer, user_data):
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    now = time.time()
    elapsed = now - user_data.last_time
    if elapsed > 0:
        user_data.fps_history.append(1.0 / elapsed)
    user_data.last_time = now
    
    sink_pad = pad.get_static_pad("sink")
    format, width, height = get_caps_from_pad(sink_pad)
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    object_counts = {}
    detection_lines = []

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        
        if label in user_data.excluded_labels:
            continue
        
        if confidence > 0.4:
            object_counts[label] = object_counts.get(label, 0) + 1
            detection_lines.append(f"  {label.capitalize()} - Confidence: {confidence:.2f}")
            user_data.confidence_history[label].append(confidence)

    if object_counts:
        total = sum(object_counts.values())
        summary = ", ".join(f"{count}x {label}" for label, count in sorted(object_counts.items()))
        print(f"[Frame {user_data.get_count()}] {total} object(s): {summary}")
        for line in detection_lines:
            print(line)
    else:
        print(f"[Frame {user_data.get_count()}] No detections above threshold.")

    # Save graph every 100 frames so it always exists on disk
    if user_data.get_count() % 500 == 0 and user_data.confidence_history:
        user_data.plot_confidence()
        print(f"[Frame {user_data.get_count()}] Graph auto-saved.")
    # Ends after 1000 frames
    if user_data.get_count() >= 500:
        print("Reached 1000 frames, stopping and saving graph...")
        user_data.plot_confidence()
        os.kill(os.getpid(), signal.SIGTERM)

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class()

    # Plot on Ctrl+C
    def handle_exit(sig, frame):
        print("\nStopping... generating confidence graph.")
        user_data.plot_confidence()
        exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()

    # Also plot if app exits normally
    user_data.plot_confidence()
