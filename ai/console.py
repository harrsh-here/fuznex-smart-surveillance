"""Optional Windows Tkinter viewer for a ``CombinedPipeline``. Not used on Kaggle."""
from __future__ import annotations

import argparse
import threading
import time
import uuid


def launch_camera(pipeline, camera: int = 0, camera_id: str = "camera-01") -> None:
    """Run a small local-only preview. Importing this module has no GUI dependency."""
    import cv2
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk(); root.title("FuzNex Combined AI · Identity Disabled")
    status = tk.StringVar(value="Stopped")
    ttk.Label(root, textvariable=status, padding=12).pack()
    running = threading.Event(); running.set()

    def worker() -> None:
        cap = cv2.VideoCapture(camera); epoch = uuid.uuid4().hex; sequence = 0
        try:
            while running.is_set():
                ok, frame = cap.read()
                if not ok:
                    status.set("Camera unavailable; stale output discarded")
                    break
                result = pipeline.process_frame(frame, camera_id, epoch, sequence, acquired_at=time.perf_counter())
                sequence += 1
                status.set("Identity disabled | %d detections | %d errors" %
                           (len(result["detections"]), len(result["processing_errors"])))
        finally:
            cap.release()

    ttk.Button(root, text="Start", command=lambda: threading.Thread(target=worker, daemon=True).start()).pack()
    ttk.Button(root, text="Stop", command=lambda: running.clear()).pack()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.parse_args()
    raise SystemExit("Construct a CombinedPipeline with explicit trusted model paths before launching this viewer.")
