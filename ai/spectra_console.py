"""The local, camera-first FuzNex SPECTRA AI Diagnostic Console.

This is intentionally a single-process Tk application so it stays portable on
Windows CPU/GPU machines.  Models, configuration, gallery, captures and logs
live below ``local_data/`` (which is ignored by Git).  The legacy CLI modules
remain supported for headless/Kaggle use.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.combined import CombinedPipeline
from ai.console import EmptyFacePipeline, annotate
from ai.config import SFaceConfig
from ai.detection import SCRFD500MDetector
from ai.enrollment import quality_embedding
from ai.local_gallery import (delete_participant, new_gallery, parse_gallery,
                              sha256_file, update_participant, write_gallery_safely)
from ai.object_detection import YoloConfig, YoloPersonLaptopDetector
from ai.pipeline import SFacePipeline
from ai.recognition import SFaceRecognizer, enrolled_embedding
from ai.spectra_diag import DiagnosticExport, GroundTruth, RunMetrics

APP_DIR = Path(__file__).resolve().parent.parent
LOCAL_DIR = APP_DIR / "local_data" / "spectra"
CONFIG_PATH = LOCAL_DIR / "console-config.json"
GALLERY_PATH = LOCAL_DIR / "private-gallery.json"


def _hash(path: str) -> str | None:
    try: return sha256_file(Path(path))
    except (OSError, ValueError): return None


def _read_config() -> dict[str, Any]:
    try: return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {"camera": "0", "samples": 5}


def _write_config(data: dict[str, Any]) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class FrameMessage:
    frame: Any | None = None
    result: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    status: str | None = None


class CaptureWorker(threading.Thread):
    """One bounded camera loop; inference never runs on Tk's event loop."""
    def __init__(self, pipeline: CombinedPipeline, source: str, output: queue.Queue[FrameMessage],
                 stop: threading.Event, exporter: DiagnosticExport | None, truth: GroundTruth | None = None) -> None:
        super().__init__(daemon=True); self.pipeline = pipeline; self.source = source
        self.output, self.stop, self.exporter, self.truth = output, stop, exporter, truth; self.metrics = RunMetrics()

    def _send(self, message: FrameMessage) -> None:
        try: self.output.put_nowait(message)
        except queue.Full: self.metrics.dropped_frames += 1

    def run(self) -> None:
        import cv2
        source = int(self.source) if self.source.isdecimal() else self.source
        cap = None; epoch = uuid.uuid4().hex; seq = 0
        try:
            while not self.stop.is_set():
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        self.metrics.errors += 1; self._send(FrameMessage(status="Camera unavailable — retrying")); time.sleep(1); continue
                    if seq:
                        self.metrics.reconnects += 1
                        reset = getattr(self.pipeline.face_pipeline, "reset_tracking", None)
                        if reset: reset()  # no stale private identity crosses a reconnect
                    self._send(FrameMessage(status="Camera connected"))
                acquired = time.perf_counter(); ok, frame = cap.read()
                self.metrics.capture_ms.append((time.perf_counter() - acquired) * 1000)
                if not ok:
                    self.metrics.dropped_frames += 1; cap.release(); cap = None
                    self._send(FrameMessage(status="Frame lost — reconnecting")); continue
                result = self.pipeline.process_frame(frame, f"source:{self.source}", epoch, seq, acquired_at=acquired); seq += 1
                self.metrics.frames += 1; process = result["processing"]
                self.metrics.inference_ms.append(process["total_ms"]); self.metrics.end_to_end_ms.append(process["capture_to_output_ms"])
                self.metrics.face_ms.append(process.get("face_ms", 0)); self.metrics.object_ms.append(process.get("object_ms", 0))
                self.metrics.errors += len(result["processing_errors"])
                if self.exporter: self.exporter.record(result, self.metrics, self.truth)
                self._send(FrameMessage(frame=frame, result=result, metrics=self.metrics.snapshot()))
        finally:
            if cap is not None: cap.release()
            if self.exporter: self.exporter.close()


class SpectraConsole:
    BG, PANEL, PANEL_2, TEXT, MUTED, ACCENT, GREEN, AMBER, RED = ("#0d1421", "#141f30", "#1b2a40", "#f4f7fb", "#9bb0c9", "#60a5fa", "#42d39d", "#f5b942", "#f87171")
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk; self.root = root; self.config = _read_config()
        self.messages: queue.Queue[FrameMessage] = queue.Queue(maxsize=2); self.stop = threading.Event(); self.worker: CaptureWorker | None = None
        self.pipeline: CombinedPipeline | None = None; self.last_frame: Any = None; self.enrolling = False; self.samples: list[Any] = []; self.blurs: list[float] = []
        self.visible = {"face", "person", "laptop"}; self.session_export: DiagnosticExport | None = None
        self._style(); self._build(); self._refresh_people(); self._log("Console ready. Select local models once, then start a live test.")
        self.root.after(30, self._poll); self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self) -> None:
        self.root.title("FuzNex · SPECTRA AI Diagnostic Console"); self.root.geometry("1450x920"); self.root.minsize(1120, 720); self.root.configure(bg=self.BG)
        s = self.ttk.Style(self.root); s.theme_use("clam")
        s.configure("TFrame", background=self.BG); s.configure("Panel.TFrame", background=self.PANEL)
        s.configure("TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 10))
        s.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI Semibold", 25))
        s.configure("Sub.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 10))
        s.configure("Metric.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI Semibold", 18))
        s.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 8)); s.map("TButton", background=[("active", self.PANEL_2)])
        s.configure("TEntry", fieldbackground="#0a101b", foreground=self.TEXT, insertcolor=self.TEXT, padding=7)
        s.configure("TCheckbutton", background=self.PANEL, foreground=self.TEXT); s.configure("TNotebook", background=self.BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=self.PANEL, foreground=self.MUTED, padding=(18, 10), font=("Segoe UI Semibold", 10)); s.map("TNotebook.Tab", background=[("selected", self.PANEL_2)], foreground=[("selected", self.TEXT)])

    def _card(self, parent: Any, **pack: Any) -> Any:
        box = self.ttk.Frame(parent, style="Panel.TFrame"); box.pack(**pack); return box

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        header = ttk.Frame(self.root); header.pack(fill="x", padx=30, pady=(22, 8))
        ttk.Label(header, text="SPECTRA AI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Private local diagnostic console  ·  camera-first testing  ·  identity disabled by default", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))
        self.status = tk.StringVar(value="Ready"); ttk.Label(self.root, textvariable=self.status, style="Sub.TLabel").pack(anchor="w", padx=32, pady=(0, 12))
        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill="both", expand=True, padx=30, pady=(0, 24))
        self.live = ttk.Frame(self.tabs); self.enroll = ttk.Frame(self.tabs); self.people = ttk.Frame(self.tabs); self.diag = ttk.Frame(self.tabs); self.logs = ttk.Frame(self.tabs)
        self.tabs.add(self.live, text="Live test"); self.tabs.add(self.enroll, text="Enroll person"); self.tabs.add(self.people, text="People"); self.tabs.add(self.diag, text="Diagnostics"); self.tabs.add(self.logs, text="Logs & sessions")
        self._build_live(); self._build_enroll(); self._build_people(); self._build_diag(); self._build_logs()

    def _build_live(self) -> None:
        tk, ttk = self.tk, self.ttk
        controls = self._card(self.live, fill="x", pady=(0, 12)); self.camera = tk.StringVar(value=str(self.config.get("camera", "0")))
        ttk.Label(controls, text="Camera / stream").pack(side="left", padx=(16, 6), pady=14); ttk.Entry(controls, textvariable=self.camera, width=30).pack(side="left", pady=14)
        for kind in ("face", "person", "laptop"):
            value = tk.BooleanVar(value=True); ttk.Checkbutton(controls, text=("SCRFD faces" if kind == "face" else f"YOLO {kind}s"), variable=value, command=lambda k=kind, v=value: self._toggle(k, v.get())).pack(side="left", padx=8)
        self.recognition = tk.BooleanVar(value=False); ttk.Checkbutton(controls, text="Enable local recognition", variable=self.recognition, command=self._recognition_changed).pack(side="left", padx=8)
        ttk.Button(controls, text="Start live test", command=self.start).pack(side="right", padx=8); ttk.Button(controls, text="Stop", command=self.halt).pack(side="right")
        body = ttk.Frame(self.live); body.pack(fill="both", expand=True)
        self.preview = tk.Label(body, text="Select models in Diagnostics, then start a camera test", bg="#080d16", fg=self.MUTED, font=("Segoe UI", 13), anchor="center")
        self.preview.pack(side="left", fill="both", expand=True)
        side = self._card(body, side="right", fill="y", padx=(14, 0)); side.configure(width=275); side.pack_propagate(False)
        ttk.Label(side, text="LIVE COUNTS", font=("Segoe UI Semibold", 11)).pack(anchor="w", padx=18, pady=(20, 8))
        self.counts = {k: tk.StringVar(value="0") for k in ("face", "person", "laptop")}
        for k in self.counts: ttk.Label(side, textvariable=self.counts[k], style="Metric.TLabel").pack(anchor="w", padx=18); ttk.Label(side, text=k.title()).pack(anchor="w", padx=19, pady=(0, 11))
        self.health = tk.StringVar(value="Awaiting live test"); ttk.Separator(side).pack(fill="x", padx=18, pady=8); ttk.Label(side, text="HEALTH", font=("Segoe UI Semibold", 11)).pack(anchor="w", padx=18); ttk.Label(side, textvariable=self.health, wraplength=230, justify="left").pack(anchor="w", padx=18, pady=8)

    def _build_enroll(self) -> None:
        tk, ttk = self.tk, self.ttk; box = self._card(self.enroll, fill="x", pady=(0, 12)); self.name = tk.StringVar(); self.consent = tk.BooleanVar(value=False); self.enroll_progress = tk.StringVar(value="Enter a display name and opt in. Nothing is retained until the final save.")
        ttk.Label(box, text="Display name").grid(row=0, column=0, padx=(18, 6), pady=(18, 8), sticky="w"); ttk.Entry(box, textvariable=self.name, width=35).grid(row=0, column=1, padx=6, pady=(18, 8), sticky="w")
        ttk.Checkbutton(box, text="I have this person's informed consent for local-only testing", variable=self.consent).grid(row=1, column=0, columnspan=3, padx=18, pady=6, sticky="w")
        self.enroll_button = ttk.Button(box, text="Begin guided captures", command=self.begin_enrollment); self.enroll_button.grid(row=2, column=0, padx=18, pady=(8, 18), sticky="w"); ttk.Button(box, text="Cancel", command=self.cancel_enrollment).grid(row=2, column=1, pady=(8, 18), sticky="w")
        help_box = self._card(self.enroll, fill="both", expand=True); ttk.Label(help_box, text="Guided automatic capture", font=("Segoe UI Semibold", 16)).pack(anchor="w", padx=22, pady=(22, 8)); ttk.Label(help_box, text="Keep exactly one face in view. The console automatically accepts sharp, well-sized samples from the live camera. It explains rejected frames and stores only the averaged template after enough approved samples. No face photos are kept.", wraplength=720, justify="left").pack(anchor="w", padx=22); ttk.Label(help_box, textvariable=self.enroll_progress, foreground=self.ACCENT, wraplength=720, justify="left").pack(anchor="w", padx=22, pady=20)

    def _build_people(self) -> None:
        tk, ttk = self.tk, self.ttk; bar = self._card(self.people, fill="x", pady=(0, 12)); self.search = tk.StringVar(); self.search.trace_add("write", lambda *_: self._refresh_people())
        ttk.Label(bar, text="Search private gallery").pack(side="left", padx=(16, 6), pady=12); ttk.Entry(bar, textvariable=self.search, width=36).pack(side="left", pady=12); ttk.Button(bar, text="Refresh", command=self._refresh_people).pack(side="right", padx=10)
        self.people_list = tk.Listbox(self.people, bg="#080d16", fg=self.TEXT, selectbackground=self.ACCENT, activestyle="none", font=("Segoe UI", 11), borderwidth=0); self.people_list.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Button(self.people, text="Remove selected person", command=self.remove_selected).pack(anchor="e")

    def _build_diag(self) -> None:
        tk, ttk = self.tk, self.ttk; box = self._card(self.diag, fill="x"); self.yolo = tk.StringVar(value=str(self.config.get("yolo", ""))); self.scrfd = tk.StringVar(value=str(self.config.get("scrfd", ""))); self.sface = tk.StringVar(value=str(self.config.get("sface", "")))
        for row, (label, variable) in enumerate((("YOLO weights (person/laptop)", self.yolo), ("SCRFD face model", self.scrfd), ("SFace recognition model", self.sface))):
            ttk.Label(box, text=label).grid(row=row, column=0, padx=(18, 8), pady=10, sticky="w"); self.ttk.Entry(box, textvariable=variable, width=72).grid(row=row, column=1, pady=10, sticky="ew"); self.ttk.Button(box, text="Choose…", command=lambda v=variable: self._choose_model(v)).grid(row=row, column=2, padx=10, pady=10)
        self.model_status = tk.StringVar(value="Models are local-only. Select available files; hashes are verified before use."); self.ttk.Label(box, textvariable=self.model_status, foreground=self.MUTED, wraplength=850).grid(row=3, column=0, columnspan=3, padx=18, pady=(5, 14), sticky="w"); self.ttk.Button(box, text="Verify & save local setup", command=self.verify_setup).grid(row=4, column=0, padx=18, pady=(0, 18), sticky="w")
        metrics = self._card(self.diag, fill="x", pady=14); self.diagnostic_text = tk.StringVar(value="No live metrics yet."); self.ttk.Label(metrics, text="Real-time diagnostic summary", font=("Segoe UI Semibold", 14)).pack(anchor="w", padx=18, pady=(16, 4)); self.ttk.Label(metrics, textvariable=self.diagnostic_text, wraplength=950, justify="left").pack(anchor="w", padx=18, pady=(0, 16))

    def _build_logs(self) -> None:
        tk, ttk = self.tk, self.ttk; controls = self._card(self.logs, fill="x"); self.distance = tk.StringVar(value="unspecified"); self.capture_opt_in = tk.BooleanVar(value=False); self.truth_flags = {k: tk.BooleanVar(value=False) for k in ("face", "person", "laptop")}
        ttk.Label(controls, text="Exploratory distance").pack(side="left", padx=(16, 6), pady=12); ttk.Combobox(controls, textvariable=self.distance, values=("unspecified", "near", "middle", "far"), width=15, state="readonly").pack(side="left", pady=12)
        for kind, variable in self.truth_flags.items(): ttk.Checkbutton(controls, text=f"Truth: {kind}", variable=variable).pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="Consent-gated captures (not implemented: no images retained)", variable=self.capture_opt_in).pack(side="left", padx=12); ttk.Button(controls, text="Export new session", command=self.new_session).pack(side="right", padx=10)
        self.log_view = tk.Text(self.logs, bg="#080d16", fg="#d7e3f4", insertbackground=self.TEXT, font=("Cascadia Mono", 9), borderwidth=0, state="disabled", wrap="word"); self.log_view.pack(fill="both", expand=True, pady=(12, 0))

    def _choose_model(self, variable: Any) -> None:
        from tkinter import filedialog
        chosen = filedialog.askopenfilename(title="Choose local model", filetypes=[("Model files", "*.onnx *.pt"), ("All files", "*.*")])
        if chosen: variable.set(chosen)

    def verify_setup(self) -> None:
        values = {"yolo": self.yolo.get().strip(), "scrfd": self.scrfd.get().strip(), "sface": self.sface.get().strip(), "camera": self.camera.get().strip(), "samples": 5}
        present = {key: value for key, value in values.items() if key in {"yolo", "scrfd", "sface"} and value}
        missing = [key.upper() for key, value in present.items() if not Path(value).is_file()]
        if missing: self.model_status.set("Missing or unreadable: " + ", ".join(missing)); return
        values["hashes"] = {key: _hash(value) for key, value in present.items()}; _write_config(values)
        self.config = values; self.model_status.set("Saved local setup · " + " · ".join(f"{k.upper()} SHA-256 {v[:12]}…" for k, v in values["hashes"].items()) if present else "Saved camera setup. Add model files for inference."); self._log("Local model setup verified and saved (paths remain in ignored local_data).")

    def _make_pipeline(self) -> CombinedPipeline:
        yolo_path, scrfd_path, sface_path = self.yolo.get().strip(), self.scrfd.get().strip(), self.sface.get().strip()
        if not yolo_path or not Path(yolo_path).is_file(): raise ValueError("Select a readable YOLO model in Diagnostics before live testing")
        objects = YoloPersonLaptopDetector(YoloConfig(model_path=Path(yolo_path)))
        faces: Any = EmptyFacePipeline()
        if scrfd_path and Path(scrfd_path).is_file():
            if self.recognition.get():
                if not sface_path or not Path(sface_path).is_file(): raise ValueError("Local recognition needs a readable SFace model")
                templates = {}
                if GALLERY_PATH.exists():
                    from ai.local_gallery import templates_from_gallery
                    templates = templates_from_gallery(GALLERY_PATH, sha256_file(Path(sface_path)))
                cfg = SFaceConfig(Path(scrfd_path), Path(sface_path)); faces = SFacePipeline(cfg, SCRFD500MDetector(cfg.detector_model), SFaceRecognizer(cfg.recognizer_model), templates)
            else:
                # SCRFD-only status uses no recognition model or local template.
                detector = SCRFD500MDetector(Path(scrfd_path))
                class DetectionOnly:
                    def process_frame(self, frame: Any) -> Any:
                        rows = []
                        for box, _landmarks in detector.detect(frame):
                            rows.append(type("Face", (), {"bbox_xyxy": box[:4], "detection_score": float(box[4]), "state": "detection_only", "quality_reason": None, "score_age_ms": 0})())
                        return type("Faces", (), {"faces": tuple(rows)})()
                faces = DetectionOnly()
        return CombinedPipeline(faces, objects)

    def start(self) -> None:
        if self.worker and self.worker.is_alive(): return
        try: self.pipeline = self._make_pipeline()
        except Exception as exc: self._error(str(exc)); return
        truth = self._ground_truth() if self.session_export else None
        self.stop.clear(); self.worker = CaptureWorker(self.pipeline, self.camera.get().strip() or "0", self.messages, self.stop, self.session_export, truth); self.worker.start(); self.status.set("Starting live test…"); self._log("Live test started; identity publication remains disabled in standard output.")

    def halt(self) -> None:
        self.stop.set(); self.status.set("Stopping live test…"); self.enrolling = False

    def _toggle(self, kind: str, enabled: bool) -> None:
        (self.visible.add if enabled else self.visible.discard)(kind)

    def _recognition_changed(self) -> None:
        if self.recognition.get(): self._log("Local recognition requested. It will load only the private gallery after the next start; no identities are exported.")

    def _poll(self) -> None:
        try:
            while True:
                msg = self.messages.get_nowait()
                if msg.status: self.status.set(msg.status); self._log(msg.status); continue
                if msg.frame is not None and msg.result is not None: self._render(msg); self._enrollment_frame(msg.frame)
        except queue.Empty: pass
        self.root.after(30, self._poll)

    def _render(self, msg: FrameMessage) -> None:
        import cv2
        result, metrics = msg.result, msg.metrics or {}; self.last_frame = msg.frame
        for kind in self.counts: self.counts[kind].set(str(sum(d["kind"] == kind for d in result["detections"])))
        self.health.set(f"{metrics.get('fps') or '—'} FPS\n{metrics.get('end_to_end_ms_avg') or '—'} ms end-to-end\nDropped {metrics.get('dropped_frames', 0)} · errors {metrics.get('errors', 0)}\nReconnects {metrics.get('reconnects', 0)}")
        self.diagnostic_text.set(f"Capture {metrics.get('capture_ms_avg') or '—'} ms · inference {metrics.get('inference_ms_avg') or '—'} ms · face {metrics.get('face_ms_avg') or '—'} ms · YOLO {metrics.get('object_ms_avg') or '—'} ms · dropped {metrics.get('dropped_frames', 0)} · errors {metrics.get('errors', 0)}")
        image = cv2.cvtColor(annotate(msg.frame.copy(), result, self.visible), cv2.COLOR_BGR2RGB); h, w = image.shape[:2]; scale = min(1040 / w, 700 / h, 1); image = cv2.resize(image, (int(w * scale), int(h * scale)))
        photo = self.tk.PhotoImage(data=(f"P6 {image.shape[1]} {image.shape[0]} 255\n".encode() + image.tobytes()), format="PPM"); self.preview.configure(image=photo, text=""); self.preview.image = photo

    def begin_enrollment(self) -> None:
        if not self.name.get().strip(): self._error("Enter a display name first."); return
        if not self.consent.get(): self._error("Confirm informed consent before enrollment."); return
        if not self.worker or not self.worker.is_alive(): self._error("Start the live camera first; guided captures use that preview."); return
        if not self.scrfd.get().strip() or not self.sface.get().strip(): self._error("Enrollment needs SCRFD and SFace models selected in Diagnostics."); return
        self.enrolling = True; self.samples.clear(); self.blurs.clear(); self.enroll_progress.set("Looking for one sharp, well-sized face…")

    def cancel_enrollment(self) -> None:
        self.enrolling = False; self.samples.clear(); self.blurs.clear(); self.enroll_progress.set("Enrollment cancelled. Nothing was saved.")

    def _enrollment_frame(self, frame: Any) -> None:
        if not self.enrolling: return
        try:
            detector = getattr(self, "_enrollment_detector", None); recognizer = getattr(self, "_enrollment_recognizer", None)
            if detector is None:
                self._enrollment_detector = detector = SCRFD500MDetector(Path(self.scrfd.get().strip())); self._enrollment_recognizer = recognizer = SFaceRecognizer(Path(self.sface.get().strip()))
            vector, blur = quality_embedding(frame, detector, recognizer); self.samples.append(vector); self.blurs.append(blur)
            needed = 5 - len(self.samples); self.enroll_progress.set(f"Accepted {len(self.samples)}/5 high-quality samples · sharpness {blur:.0f}" + (". Keep steady…" if needed else ". Saving private template…"))
            if not needed: self._save_enrollment()
        except ValueError as exc: self.enroll_progress.set(f"Waiting: {exc}")
        except Exception as exc: self.enrolling = False; self._error(f"Enrollment stopped: {exc}")

    def _save_enrollment(self) -> None:
        try:
            model_hash = sha256_file(Path(self.sface.get().strip())); embedding = enrolled_embedding(self.samples)
            gallery = parse_gallery(GALLERY_PATH, model_hash) if GALLERY_PATH.exists() else new_gallery(model_hash, int(embedding.size))
            person_id = "local-" + uuid.uuid4().hex[:12]; quality = {"min_face_size_px": 80, "min_blur": 40, "observed_blur_min": round(min(self.blurs), 2), "observed_blur_max": round(max(self.blurs), 2)}
            write_gallery_safely(GALLERY_PATH, update_participant(gallery, person_id, embedding, len(self.samples), quality, self.name.get().strip()))
            self.enrolling = False; self.enroll_progress.set(f"Saved {self.name.get().strip()} locally with {len(self.samples)} quality-approved samples. No images were retained."); self._refresh_people(); self._log("A consented local template was enrolled; its name and embedding remain in the ignored private gallery.")
        except Exception as exc: self.enrolling = False; self._error(f"Could not save enrollment: {exc}")

    def _refresh_people(self) -> None:
        self.people_list.delete(0, "end"); self._people: list[dict[str, Any]] = []
        try: gallery = parse_gallery(GALLERY_PATH) if GALLERY_PATH.exists() else {"participants": []}
        except ValueError as exc: self._error(f"Private gallery cannot be read: {exc}"); return
        needle = getattr(self, "search", self.tk.StringVar(value="")).get().lower()
        for entry in gallery["participants"]:
            if needle in entry["display_name"].lower() or needle in entry["participant_id"].lower():
                self._people.append(entry); self.people_list.insert("end", f"{entry['display_name']}   ·   {entry['sample_count']} samples   ·   {entry['participant_id']}")

    def remove_selected(self) -> None:
        from tkinter import messagebox
        selected = self.people_list.curselection()
        if not selected: self._error("Select a person to remove."); return
        entry = self._people[selected[0]]
        if not messagebox.askyesno("Remove private template", f"Remove the local template for {entry['display_name']}? A timestamped backup will be retained locally."): return
        try:
            gallery = parse_gallery(GALLERY_PATH); write_gallery_safely(GALLERY_PATH, delete_participant(gallery, entry["participant_id"])); self._refresh_people(); self._log("A local private template was removed; a local backup was created.")
        except Exception as exc: self._error(f"Could not remove template: {exc}")

    def _ground_truth(self) -> GroundTruth:
        """Optional manual truth for the current export; never includes a person identity."""
        band = self.distance.get()
        return GroundTruth(frozenset(kind for kind, flag in self.truth_flags.items() if flag.get()),
                           None if band == "unspecified" else band)

    def new_session(self) -> None:
        if self.worker and self.worker.is_alive(): self._error("Stop the current live test before starting a new export session."); return
        provenance = {"models": {k: v for k, v in {"yolo_sha256": _hash(self.yolo.get().strip()) if self.yolo.get().strip() else None, "scrfd_sha256": _hash(self.scrfd.get().strip()) if self.scrfd.get().strip() else None}.items() if v}, "distance": self.distance.get(), "privacy": {"identity_publication": "disabled", "face_person_association": "disabled", "captures": "not_retained"}}
        try: self.session_export = DiagnosticExport(APP_DIR / "diagnostics", provenance); self._log(f"New non-overwriting JSONL/CSV session prepared: {self.session_export.directory}"); self.status.set("Export session ready; start live test to record it.")
        except FileExistsError: self._error("Could not allocate a unique export directory; please retry.")

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S"); self.log_view.configure(state="normal"); self.log_view.insert("end", f"[{stamp}] {text}\n"); self.log_view.see("end"); self.log_view.configure(state="disabled")

    def _error(self, text: str) -> None:
        self.status.set(text); self._log("ERROR: " + text)

    def close(self) -> None:
        self.halt(); self.root.destroy()


def main() -> int:
    import tkinter as tk
    root = tk.Tk(); SpectraConsole(root); root.mainloop(); return 0


if __name__ == "__main__": raise SystemExit(main())
