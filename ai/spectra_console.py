"""A task-first, local-only desktop console for FuzNex SPECTRA diagnostics.

The UI deliberately keeps setup and model paths out of the everyday camera,
people, diagnostics, and export flows.  Local configuration, templates and
exports remain ignored by Git under ``local_data/`` and ``diagnostics/``.
"""
from __future__ import annotations

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
                              sha256_file, templates_from_gallery,
                              update_participant, write_gallery_safely)
from ai.object_detection import YoloConfig, YoloPersonLaptopDetector
from ai.pipeline import SFacePipeline
from ai.recognition import SFaceRecognizer, enrolled_embedding
from ai.runtime_discovery import discover_cameras, discover_models
from ai.spectra_diag import DiagnosticExport, GroundTruth, RunMetrics

APP_DIR = Path(__file__).resolve().parent.parent
LOCAL_DIR = APP_DIR / "local_data" / "spectra"
CONFIG_PATH = LOCAL_DIR / "console-config.json"
GALLERY_PATH = LOCAL_DIR / "private-gallery.json"


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_config(config: dict[str, Any]) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _model_hash(path: str) -> str | None:
    try:
        return sha256_file(Path(path))
    except OSError:
        return None


@dataclass
class FrameMessage:
    frame: Any | None = None
    result: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    status: str | None = None


class CaptureWorker(threading.Thread):
    """Bounded capture/inference loop that never blocks the Tk event thread."""
    def __init__(self, pipeline: CombinedPipeline, source: str, inbox: queue.Queue[FrameMessage],
                 stop: threading.Event, exporter: DiagnosticExport | None, truth: GroundTruth | None) -> None:
        super().__init__(daemon=True)
        self.pipeline, self.source, self.inbox, self.stop = pipeline, source, inbox, stop
        self.exporter, self.truth, self.metrics = exporter, truth, RunMetrics()

    def _offer(self, message: FrameMessage) -> None:
        try:
            self.inbox.put_nowait(message)
        except queue.Full:
            self.metrics.dropped_frames += 1

    def run(self) -> None:
        import cv2
        source = int(self.source) if self.source.isdecimal() else self.source
        cap = None; epoch = uuid.uuid4().hex; sequence = 0
        try:
            while not self.stop.is_set():
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        self.metrics.errors += 1
                        self._offer(FrameMessage(status="error|Camera unavailable. Check the connection and try again."))
                        time.sleep(1); continue
                    if sequence:
                        self.metrics.reconnects += 1
                        reset = getattr(self.pipeline.face_pipeline, "reset_tracking", None)
                        if reset: reset()
                    self._offer(FrameMessage(status="info|Camera connected."))
                acquired = time.perf_counter(); ok, frame = cap.read()
                self.metrics.capture_ms.append((time.perf_counter() - acquired) * 1000)
                if not ok:
                    self.metrics.dropped_frames += 1; cap.release(); cap = None
                    self._offer(FrameMessage(status="warning|A frame was lost. Reconnecting…")); continue
                result = self.pipeline.process_frame(frame, f"source:{self.source}", epoch, sequence, acquired_at=acquired); sequence += 1
                self.metrics.frames += 1; processing = result["processing"]
                self.metrics.inference_ms.append(processing["total_ms"]); self.metrics.end_to_end_ms.append(processing["capture_to_output_ms"])
                self.metrics.face_ms.append(processing.get("face_ms", 0)); self.metrics.object_ms.append(processing.get("object_ms", 0))
                self.metrics.errors += len(result["processing_errors"])
                if result["processing_errors"]:
                    self._offer(FrameMessage(status="error|One detection branch could not process this frame. Live test continues."))
                if self.exporter: self.exporter.record(result, self.metrics, self.truth)
                self._offer(FrameMessage(frame=frame, result=result, metrics=self.metrics.snapshot()))
        finally:
            if cap is not None: cap.release()
            if self.exporter: self.exporter.close()


class SpectraConsole:
    """Four-task desktop shell with a deliberately quiet first-run experience."""
    BG = "#09111f"; SIDE = "#0d1828"; SURFACE = "#111f32"; SURFACE_2 = "#172a42"
    TEXT = "#f1f5f9"; MUTED = "#9eb1c8"; BLUE = "#4f9cf9"; GREEN = "#35c58b"; AMBER = "#f4b844"; RED = "#fb7185"

    def __init__(self, root: Any) -> None:
        import tkinter as tk
        self.tk = tk; self.root = root; self.config = _load_config()
        self.inbox: queue.Queue[FrameMessage] = queue.Queue(maxsize=2); self.runtime_inbox: queue.Queue[tuple[dict[str, Any], dict[str, str], list[str], str | None]] = queue.Queue(maxsize=1); self.stop = threading.Event(); self.worker: CaptureWorker | None = None
        self.pipeline: CombinedPipeline | None = None; self.last_metrics: dict[str, Any] = {}; self.last_frame: Any = None; self.exporter: DiagnosticExport | None = None
        self.enrolling = False; self.samples: list[Any] = []; self.blurs: list[float] = []; self.events: list[tuple[str, str, str]] = []; self.model_availability: dict[str, str] = {}; self.camera_candidates: list[str] = []; self.active_camera: str | None = None; self.last_detection_signature: tuple[tuple[str, int], ...] | None = None
        self.show_faces = True; self.show_persons = True; self.show_laptops = True; self.show_phones = True; self.recognition_enabled = False
        self._style(); self._build_shell(); self.show_page("live"); self._refresh_people(); self._event("info", "Console ready", "Start a camera test or open setup if this is your first use.")
        self._discover_runtime(); self.root.after(30, self._poll); self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self) -> None:
        self.root.title("FuzNex · SPECTRA AI Diagnostic Console"); self.root.geometry("1440x900"); self.root.minsize(1080, 680); self.root.configure(bg=self.BG)
        self.root.option_add("*Font", "{Segoe UI} 10"); self.root.option_add("*Button.Cursor", "hand2")

    def _button(self, parent: Any, text: str, command: Any, primary: bool = False, danger: bool = False, width: int | None = None) -> Any:
        bg = self.RED if danger else self.BLUE if primary else self.SURFACE_2
        button = self.tk.Button(parent, text=text, command=command, bg=bg, fg="#ffffff", activebackground=bg,
                                activeforeground="#ffffff", relief="flat", bd=0, padx=16, pady=10,
                                font=("Segoe UI Semibold", 10), takefocus=True, width=width)
        return button

    def _label(self, parent: Any, text: str = "", *, size: int = 10, bold: bool = False, color: str | None = None, **kwargs: Any) -> Any:
        return self.tk.Label(parent, text=text, bg=kwargs.pop("bg", self.SURFACE), fg=color or self.TEXT,
                             font=("Segoe UI Semibold" if bold else "Segoe UI", size), **kwargs)

    def _card(self, parent: Any, **kwargs: Any) -> Any:
        return self.tk.Frame(parent, bg=self.SURFACE, highlightbackground="#233a58", highlightthickness=1, **kwargs)

    def _build_shell(self) -> None:
        tk = self.tk
        shell = tk.Frame(self.root, bg=self.BG); shell.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(shell, bg=self.SIDE, width=236); self.sidebar.pack(side="left", fill="y"); self.sidebar.pack_propagate(False)
        brand = tk.Frame(self.sidebar, bg=self.SIDE); brand.pack(fill="x", padx=22, pady=(26, 30))
        self._label(brand, "SPECTRA", size=21, bold=True, bg=self.SIDE).pack(anchor="w")
        self._label(brand, "AI diagnostic console", size=9, color=self.MUTED, bg=self.SIDE).pack(anchor="w", pady=(2, 0))
        self.nav: dict[str, Any] = {}
        for key, label, glyph in (("live", "Live test", "●"), ("people", "People", "●"), ("diagnostics", "Diagnostics", "●"), ("export", "Export results", "●")):
            item = tk.Button(self.sidebar, text=f"{glyph}   {label}", anchor="w", command=lambda k=key: self.show_page(k),
                             bg=self.SIDE, fg=self.MUTED, activebackground=self.SURFACE_2, activeforeground=self.TEXT,
                             relief="flat", bd=0, padx=21, pady=12, font=("Segoe UI Semibold", 10), takefocus=True)
            item.pack(fill="x", padx=10, pady=2); self.nav[key] = item
        tk.Frame(self.sidebar, bg="#203653", height=1).pack(fill="x", padx=20, pady=18)
        self._button(self.sidebar, "Advanced settings", self.open_setup, width=19).pack(padx=20, pady=2)
        self._label(self.sidebar, "Local only\nNames, templates, images and model paths never leave this device.", size=9, color=self.MUTED, bg=self.SIDE, justify="left", wraplength=184).pack(side="bottom", anchor="w", padx=22, pady=24)
        main = tk.Frame(shell, bg=self.BG); main.pack(side="left", fill="both", expand=True)
        top = tk.Frame(main, bg=self.BG, height=86); top.pack(fill="x", padx=32, pady=(20, 0)); top.pack_propagate(False)
        title_box = tk.Frame(top, bg=self.BG); title_box.pack(side="left", fill="y")
        self.page_title = self._label(title_box, bg=self.BG, size=22, bold=True); self.page_title.pack(anchor="w", pady=(7, 1))
        self.page_subtitle = self._label(title_box, bg=self.BG, size=10, color=self.MUTED); self.page_subtitle.pack(anchor="w")
        self.ready_badge = self._label(top, "", size=9, bold=True, bg=self.BG); self.ready_badge.pack(side="right", pady=21)
        self.content = tk.Frame(main, bg=self.BG); self.content.pack(fill="both", expand=True, padx=32, pady=(10, 28))
        self.pages = {"live": self._build_live(), "people": self._build_people(), "diagnostics": self._build_diagnostics(), "export": self._build_export()}

    def _page(self) -> Any:
        page = self.tk.Frame(self.content, bg=self.BG); page.place(relwidth=1, relheight=1); return page

    def _build_live(self) -> Any:
        tk = self.tk; page = self._page(); action = self._card(page); action.pack(fill="x", pady=(0, 14))
        copy = tk.Frame(action, bg=self.SURFACE); copy.pack(side="left", padx=20, pady=16)
        self._label(copy, "Camera", size=12, bold=True).pack(anchor="w"); self.live_status = self._label(copy, "Ready to start", size=10, color=self.MUTED); self.live_status.pack(anchor="w", pady=(3, 0))
        self.missing_model_button = self._button(action, "Set up missing model", self.open_setup); self.missing_model_button.pack(side="right", padx=(8, 0), pady=16); self.missing_model_button.pack_forget()
        self.start_button = self._button(action, "Start camera", self.start, primary=True); self.start_button.pack(side="right", padx=(8, 18), pady=16)
        self.stop_button = self._button(action, "Stop", self.halt); self.stop_button.pack(side="right", pady=16); self.stop_button.configure(state="disabled")
        body = tk.Frame(page, bg=self.BG); body.pack(fill="both", expand=True)
        preview_card = self._card(body); preview_card.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(preview_card, text="Your camera preview will appear here", bg="#050b14", fg=self.MUTED, font=("Segoe UI", 14), anchor="center")
        self.preview.pack(fill="both", expand=True, padx=1, pady=1)
        summary = tk.Frame(body, bg=self.BG, width=285); summary.pack(side="right", fill="y", padx=(14, 0)); summary.pack_propagate(False)
        self._live_summary(summary)
        return page

    def _live_summary(self, parent: Any) -> None:
        health = self._card(parent); health.pack(fill="x")
        self._label(health, "LIVE STATUS", size=9, bold=True, color=self.MUTED).pack(anchor="w", padx=18, pady=(17, 6))
        self.health_text = self._label(health, "Not running", size=13, bold=True); self.health_text.pack(anchor="w", padx=18)
        self.performance_text = self._label(health, "Start the camera to see performance.", size=9, color=self.MUTED, justify="left", wraplength=240); self.performance_text.pack(anchor="w", padx=18, pady=(5, 17))
        counts = self._card(parent); counts.pack(fill="x", pady=14); self._label(counts, "THIS FRAME", size=9, bold=True, color=self.MUTED).pack(anchor="w", padx=18, pady=(17, 8))
        self.count_text = self._label(counts, "0 faces\n0 people\n0 laptops\n0 phones", size=14, bold=True, justify="left", anchor="w"); self.count_text.pack(anchor="w", padx=18, pady=(0, 18))
        activity = self._card(parent); activity.pack(fill="both", expand=True); self._label(activity, "RECENT ACTIVITY", size=9, bold=True, color=self.MUTED).pack(anchor="w", padx=18, pady=(17, 8))
        self.activity_text = self._label(activity, "No events yet.", size=9, color=self.MUTED, justify="left", anchor="nw", wraplength=238); self.activity_text.pack(fill="both", padx=18, pady=(0, 16))

    def _build_people(self) -> Any:
        tk = self.tk; page = self._page(); body = tk.Frame(page, bg=self.BG); body.pack(fill="both", expand=True)
        left = self._card(body); left.pack(side="left", fill="both", expand=True)
        bar = tk.Frame(left, bg=self.SURFACE); bar.pack(fill="x", padx=18, pady=(16, 10))
        self._label(bar, "Saved people", size=12, bold=True).pack(side="left")
        self.people_search = tk.StringVar(); self.people_search.trace_add("write", lambda *_: self._refresh_people())
        search = tk.Entry(bar, textvariable=self.people_search, bg="#08101c", fg=self.TEXT, insertbackground=self.TEXT, relief="flat", width=24, font=("Segoe UI", 10)); search.pack(side="right", padx=(8, 0), ipady=6)
        self.people_list = tk.Listbox(left, bg="#08101c", fg=self.TEXT, selectbackground=self.BLUE, selectforeground="#ffffff", relief="flat", activestyle="none", font=("Segoe UI", 10), height=12, takefocus=True)
        self.people_list.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.remove_person_button = self._button(left, "Remove selected person", self.remove_selected, danger=True); self.remove_person_button.pack(anchor="e", padx=18, pady=(0, 18)); self.remove_person_button.configure(state="disabled")
        self.people_list.bind("<<ListboxSelect>>", lambda _event: self.remove_person_button.configure(state="normal") if self.people else None)
        right = self._card(body, width=390); right.pack(side="right", fill="y", padx=(14, 0)); right.pack_propagate(False)
        self._label(right, "Enroll a person", size=16, bold=True).pack(anchor="w", padx=22, pady=(22, 4))
        self._label(right, "Guided, consented and local-only. Face images are not retained.", size=9, color=self.MUTED, wraplength=334, justify="left").pack(anchor="w", padx=22, pady=(0, 20))
        self._label(right, "Display name", size=9, bold=True, color=self.MUTED).pack(anchor="w", padx=22)
        self.person_name = tk.StringVar(); entry = tk.Entry(right, textvariable=self.person_name, bg="#08101c", fg=self.TEXT, insertbackground=self.TEXT, relief="flat", font=("Segoe UI", 11)); entry.pack(fill="x", padx=22, pady=(6, 14), ipady=8)
        self.consent = tk.BooleanVar(value=False); check = tk.Checkbutton(right, text="I have this person's informed consent", variable=self.consent, bg=self.SURFACE, fg=self.TEXT, selectcolor="#08101c", activebackground=self.SURFACE, activeforeground=self.TEXT, anchor="w", justify="left", wraplength=330, takefocus=True); check.pack(fill="x", padx=18, pady=(0, 13))
        self.enroll_button = self._button(right, "Start guided enrollment", self.begin_enrollment, primary=True); self.enroll_button.pack(fill="x", padx=22)
        self._button(right, "Cancel enrollment", self.cancel_enrollment).pack(fill="x", padx=22, pady=(8, 18))
        tk.Frame(right, bg="#284463", height=1).pack(fill="x", padx=22)
        self.enroll_feedback = self._label(right, "1. Start the camera.\n2. Enter a name and confirm consent.\n3. The console collects five sharp samples automatically.", size=10, color=self.MUTED, justify="left", anchor="nw", wraplength=334); self.enroll_feedback.pack(fill="both", expand=True, padx=22, pady=18)
        return page

    def _build_diagnostics(self) -> Any:
        tk = self.tk; page = self._page(); grid = tk.Frame(page, bg=self.BG); grid.pack(fill="both", expand=True)
        overview = self._card(grid); overview.pack(fill="x")
        self._label(overview, "System health", size=16, bold=True).pack(anchor="w", padx=20, pady=(18, 3))
        self._button(overview, "Advanced detection options", self.open_detection_options).pack(anchor="e", padx=20, pady=(0, 4))
        self.diag_summary = self._label(overview, "No test is running. Set up models, then start the camera.", size=10, color=self.MUTED, wraplength=940, justify="left"); self.diag_summary.pack(anchor="w", padx=20, pady=(0, 16))
        cards = tk.Frame(grid, bg=self.BG); cards.pack(fill="x", pady=14)
        self.metric_cards: dict[str, Any] = {}
        for key, label in (("fps", "FPS"), ("latency", "End-to-end"), ("dropped", "Dropped frames"), ("errors", "Errors")):
            card = self._card(cards); card.pack(side="left", fill="both", expand=True, padx=(0 if key == "fps" else 10, 0)); self._label(card, label.upper(), size=9, bold=True, color=self.MUTED).pack(anchor="w", padx=16, pady=(15, 4)); value = self._label(card, "—", size=20, bold=True); value.pack(anchor="w", padx=16, pady=(0, 15)); self.metric_cards[key] = value
        events = self._card(grid); events.pack(fill="both", expand=True)
        bar = tk.Frame(events, bg=self.SURFACE); bar.pack(fill="x", padx=18, pady=(16, 10)); self._label(bar, "Events", size=12, bold=True).pack(side="left")
        self.log_filter = tk.StringVar(value="All"); menu = tk.OptionMenu(bar, self.log_filter, "All", "Errors", "Warnings", "Info", command=lambda *_: self._render_events()); menu.configure(bg=self.SURFACE_2, fg=self.TEXT, activebackground=self.SURFACE_2, activeforeground=self.TEXT, relief="flat", highlightthickness=0); menu["menu"].configure(bg=self.SURFACE_2, fg=self.TEXT); menu.pack(side="right")
        self.event_list = tk.Listbox(events, bg="#08101c", fg=self.TEXT, selectbackground=self.BLUE, relief="flat", font=("Cascadia Mono", 9), takefocus=True); self.event_list.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        return page

    def _build_export(self) -> Any:
        tk = self.tk; page = self._page(); card = self._card(page); card.pack(fill="x")
        self._label(card, "Save a test session", size=16, bold=True).pack(anchor="w", padx=22, pady=(22, 4))
        self._label(card, "Creates a new private JSONL and CSV diagnostic trace. Files never overwrite an earlier session and never include private names, templates or images.", size=10, color=self.MUTED, wraplength=760, justify="left").pack(anchor="w", padx=22, pady=(0, 18))
        self._button(card, "Start a saved test session", self.prepare_export, primary=True).pack(anchor="w", padx=22, pady=(0, 22))
        detail = self._card(page); detail.pack(fill="both", expand=True, pady=14)
        self._label(detail, "Optional test notes", size=12, bold=True).pack(anchor="w", padx=22, pady=(20, 3))
        self._label(detail, "Add context only when you need it. These labels describe the test, not a person.", size=9, color=self.MUTED).pack(anchor="w", padx=22, pady=(0, 14))
        line = tk.Frame(detail, bg=self.SURFACE); line.pack(fill="x", padx=22)
        self._label(line, "Distance", size=10, bold=True).pack(side="left"); self.distance = tk.StringVar(value="Not specified")
        distance = tk.OptionMenu(line, self.distance, "Not specified", "Near", "Middle", "Far"); distance.configure(bg=self.SURFACE_2, fg=self.TEXT, activebackground=self.SURFACE_2, activeforeground=self.TEXT, relief="flat", highlightthickness=0); distance["menu"].configure(bg=self.SURFACE_2, fg=self.TEXT); distance.pack(side="left", padx=10)
        self.truth_flags = {kind: tk.BooleanVar(value=False) for kind in ("face", "person", "laptop")}
        for kind, flag in self.truth_flags.items(): tk.Checkbutton(detail, text=f"Expected {kind} present", variable=flag, bg=self.SURFACE, fg=self.TEXT, selectcolor="#08101c", activebackground=self.SURFACE, activeforeground=self.TEXT, takefocus=True).pack(anchor="w", padx=22, pady=3)
        self.export_status = self._label(detail, "No session prepared.", size=10, color=self.MUTED, wraplength=700, justify="left"); self.export_status.pack(anchor="w", padx=22, pady=(18, 0))
        return page

    def show_page(self, key: str) -> None:
        titles = {"live": ("Live test", "See the camera, detections and current health in one place."), "people": ("People", "Enroll or remove consented local recognition templates."), "diagnostics": ("Diagnostics", "Review performance and clear, time-stamped events."), "export": ("Export results", "Save a new diagnostic session without exposing private identities.")}
        for name, page in self.pages.items(): page.place_forget(); self.nav[name].configure(bg=self.SIDE, fg=self.MUTED)
        self.pages[key].place(relwidth=1, relheight=1); self.nav[key].configure(bg=self.SURFACE_2, fg=self.TEXT)
        self.page_title.configure(text=titles[key][0]); self.page_subtitle.configure(text=titles[key][1]); self._set_badge()

    def _set_badge(self) -> None:
        if self.worker and self.worker.is_alive(): self.ready_badge.configure(text="●  Camera running", fg=self.GREEN)
        elif self.camera_candidates: self.ready_badge.configure(text="●  Camera ready", fg=self.GREEN)
        else: self.ready_badge.configure(text="●  Checking camera", fg=self.AMBER)

    def _setup_ready(self) -> bool:
        return bool(self.model_availability.get("yolo") == "available")

    def _discover_runtime(self) -> None:
        """Probe only known local locations off the UI thread; never downloads."""
        def probe() -> None:
            try:
                discovered = discover_models(APP_DIR, self.config)
                cameras = discover_cameras(discovered.config.get("camera"))
                self.runtime_inbox.put_nowait((discovered.config, discovered.availability, cameras, None))
            except Exception as exc:
                self.runtime_inbox.put_nowait((dict(self.config), {}, [], str(exc)))
        threading.Thread(target=probe, daemon=True).start()

    def _apply_runtime_discovery(self, config: dict[str, Any], availability: dict[str, str], cameras: list[str], error: str | None) -> None:
        if error:
            self._event("warning", "Automatic setup incomplete", "Camera or model discovery could not finish. You can still start the camera or use Advanced Settings.")
            return
        self.config, self.model_availability, self.camera_candidates = config, availability, cameras
        if cameras and self.config.get("camera") not in cameras:
            self.config["camera"] = cameras[0]
        _save_config(self.config)
        available = [name.upper() for name, state in availability.items() if state == "available"]
        missing = [name.upper() for name, state in availability.items() if state != "available"]
        self._event("info", "Local setup checked", ("Available: " + ", ".join(available) if available else "No compatible detection models found.") + (" · Missing: " + ", ".join(missing) if missing else ""))
        self._refresh_runtime_status()

    def _refresh_runtime_status(self) -> None:
        if not hasattr(self, "live_status") or (self.worker and self.worker.is_alive()): return
        capabilities = []
        if self.model_availability.get("yolo") == "available": capabilities.append("people, laptops and phones")
        if self.model_availability.get("scrfd") == "available": capabilities.append("faces")
        if self.camera_candidates:
            source = "default camera" if self.camera_candidates[0] == "0" else "saved camera"
            self.live_status.configure(text=f"Ready — {source}; detecting " + (", ".join(capabilities) if capabilities else "camera preview only"), fg=self.GREEN)
        else:
            self.live_status.configure(text="No camera was found yet. Connect one, then Start camera to retry.", fg=self.AMBER)
        if any(state != "available" for state in self.model_availability.values()): self.missing_model_button.pack(side="right", padx=(8, 0), pady=16)
        else: self.missing_model_button.pack_forget()

    def _open_file(self, variable: Any, title: str) -> None:
        from tkinter import filedialog
        selected = filedialog.askopenfilename(title=title, filetypes=[("Model files", "*.onnx *.pt"), ("All files", "*.*")])
        if selected: variable.set(selected)

    def open_setup(self) -> None:
        tk = self.tk; dialog = tk.Toplevel(self.root); dialog.title("Set up local models"); dialog.configure(bg=self.SURFACE); dialog.resizable(False, False); dialog.transient(self.root); dialog.grab_set()
        self._label(dialog, "Set up camera & local models", size=17, bold=True).pack(anchor="w", padx=24, pady=(22, 3)); self._label(dialog, "Choose your model files once. Their paths and hashes stay only on this device.", size=10, color=self.MUTED, wraplength=610, justify="left").pack(anchor="w", padx=24, pady=(0, 18))
        values = {key: tk.StringVar(value=str(self.config.get(key, ""))) for key in ("yolo", "scrfd", "sface", "camera")}
        camera_row = tk.Frame(dialog, bg=self.SURFACE); camera_row.pack(fill="x", padx=24, pady=6); self._label(camera_row, "Camera source  (0 is the default camera)", size=9, bold=True, color=self.MUTED).pack(anchor="w"); tk.Entry(camera_row, textvariable=values["camera"], bg="#08101c", fg=self.TEXT, insertbackground=self.TEXT, relief="flat", width=68).pack(fill="x", pady=(5, 0), ipady=7)
        for key, label, required in (("yolo", "YOLO person/laptop model", True), ("scrfd", "SCRFD face model", False), ("sface", "SFace recognition model", False)):
            row = tk.Frame(dialog, bg=self.SURFACE); row.pack(fill="x", padx=24, pady=6); self._label(row, label + ("  (required for default live test)" if required else "  (optional)"), size=9, bold=True, color=self.MUTED).pack(anchor="w")
            entry = tk.Entry(row, textvariable=values[key], bg="#08101c", fg=self.TEXT, insertbackground=self.TEXT, relief="flat", width=68); entry.pack(side="left", pady=(5, 0), ipady=7); self._button(row, "Choose…", lambda v=values[key], l=label: self._open_file(v, l)).pack(side="right", pady=(5, 0))
        feedback = self._label(dialog, "", size=9, color=self.MUTED, justify="left", wraplength=610); feedback.pack(anchor="w", padx=24, pady=(12, 4))
        def save() -> None:
            proposed = {key: variable.get().strip() for key, variable in values.items()}
            missing = [key.upper() for key in ("yolo", "scrfd", "sface") if proposed[key] and not Path(proposed[key]).is_file()]
            if missing: feedback.configure(text="Can't read: " + ", ".join(missing) + ". Choose the local model file again.", fg=self.RED); return
            proposed["camera"] = proposed["camera"] or "0"; proposed["hashes"] = {key: _model_hash(proposed[key]) for key in ("yolo", "scrfd", "sface") if proposed[key]}; proposed["model_dirs"] = sorted({str(Path(proposed[key]).parent) for key in ("yolo", "scrfd", "sface") if proposed[key]}); self.config = proposed; _save_config(proposed)
            feedback.configure(text="Saved locally. The camera preview works even while optional detection models are missing.", fg=self.GREEN); self._event("info", "Local setup saved", "A user-approved model location was saved for future automatic discovery."); self._discover_runtime(); self._set_badge(); dialog.after(700, dialog.destroy)
        buttons = tk.Frame(dialog, bg=self.SURFACE); buttons.pack(fill="x", padx=24, pady=(10, 22)); self._button(buttons, "Save setup", save, primary=True).pack(side="right"); self._button(buttons, "Cancel", dialog.destroy).pack(side="right", padx=8)

    def open_detection_options(self) -> None:
        tk = self.tk; dialog = tk.Toplevel(self.root); dialog.title("Detection options"); dialog.configure(bg=self.SURFACE); dialog.resizable(False, False); dialog.transient(self.root)
        self._label(dialog, "Detection options", size=16, bold=True).pack(anchor="w", padx=22, pady=(20, 3)); self._label(dialog, "These options affect the live overlay only. Identity is always hidden from standard exports.", size=9, color=self.MUTED, wraplength=460, justify="left").pack(anchor="w", padx=22, pady=(0, 12))
        options = (("Show faces", "show_faces"), ("Show people", "show_persons"), ("Show laptops", "show_laptops"), ("Show phones", "show_phones"), ("Enable private local recognition", "recognition_enabled"))
        values = {}
        for label, attr in options:
            var = tk.BooleanVar(value=getattr(self, attr)); values[attr] = var; tk.Checkbutton(dialog, text=label, variable=var, bg=self.SURFACE, fg=self.TEXT, selectcolor="#08101c", activebackground=self.SURFACE, activeforeground=self.TEXT, takefocus=True).pack(anchor="w", padx=22, pady=4)
        def apply() -> None:
            for attr, var in values.items(): setattr(self, attr, var.get())
            self._event("info", "Detection options updated", "Restart the camera for local recognition changes to take effect."); dialog.destroy()
        self._button(dialog, "Apply", apply, primary=True).pack(anchor="e", padx=22, pady=20)

    def _make_pipeline(self) -> CombinedPipeline:
        class NoObjects:
            def detect(self, _frame: Any) -> list[dict[str, Any]]: return []
        objects: Any = NoObjects()
        yolo_value = self.config.get("yolo", "")
        if yolo_value and Path(yolo_value).is_file():
            try:
                objects = YoloPersonLaptopDetector(YoloConfig(model_path=Path(yolo_value)))
                self._event("info", "Object model initialized", "YOLO is ready for people, laptops and phones.")
            except Exception as exc: self.model_availability["yolo"] = "failed"; self._event("error", "Person/laptop model unavailable", str(exc))
        faces: Any = EmptyFacePipeline(); scrfd = self.config.get("scrfd", ""); sface = self.config.get("sface", "")
        if scrfd and Path(scrfd).is_file():
            if self.recognition_enabled:
                if not sface or not Path(sface).is_file(): raise ValueError("Private recognition needs an SFace model. Open Detection options or Advanced settings.")
                templates = templates_from_gallery(GALLERY_PATH, sha256_file(Path(sface))) if GALLERY_PATH.exists() else {}
                config = SFaceConfig(Path(scrfd), Path(sface)); faces = SFacePipeline(config, SCRFD500MDetector(config.detector_model), SFaceRecognizer(config.recognizer_model), templates)
            else:
                try:
                    detector = SCRFD500MDetector(Path(scrfd))
                    self._event("info", "Face model initialized", "SCRFD is ready for local live detection.")
                except Exception as exc: self.model_availability["scrfd"] = "failed"; self._event("error", "Face model unavailable", str(exc)); return CombinedPipeline(faces, objects)
                class DetectionOnly:
                    def process_frame(self, frame: Any) -> Any:
                        rows = [type("Face", (), {"bbox_xyxy": box[:4], "detection_score": float(box[4]), "state": "detection_only", "quality_reason": None, "score_age_ms": 0})() for box, _ in detector.detect(frame)]
                        return type("Faces", (), {"faces": tuple(rows)})()
                faces = DetectionOnly()
        return CombinedPipeline(faces, objects)

    def start(self) -> None:
        if self.worker and self.worker.is_alive(): return
        try: self.pipeline = self._make_pipeline()
        except Exception as exc: self._event("error", "Camera could not start", str(exc)); return
        source = self.config.get("camera") if self.config.get("camera") in self.camera_candidates else (self.camera_candidates[0] if self.camera_candidates else "0")
        self.active_camera = str(source); self.stop.clear(); self.worker = CaptureWorker(self.pipeline, self.active_camera, self.inbox, self.stop, self.exporter, self._ground_truth() if self.exporter else None); self.worker.start()
        self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal"); self.live_status.configure(text="Starting camera…", fg=self.MUTED); self._event("info", "Live test started", "Identity publication and face-to-person association remain disabled."); self._set_badge()

    def halt(self) -> None:
        self.stop.set(); self.enrolling = False; self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled"); self.live_status.configure(text="Camera stopped", fg=self.MUTED); self.health_text.configure(text="Stopped"); self._event("info", "Live test stopped", "The camera was released. Start again whenever you are ready."); self._set_badge()

    def _poll(self) -> None:
        try:
            while True: self._apply_runtime_discovery(*self.runtime_inbox.get_nowait())
        except queue.Empty: pass
        try:
            while True:
                message = self.inbox.get_nowait()
                if message.status:
                    level, text = message.status.split("|", 1)
                    if level == "info" and text == "Camera connected." and self.active_camera is not None:
                        self.config["camera"] = self.active_camera; _save_config(self.config)
                    self._event(level, "Camera", text); continue
                if message.frame is not None and message.result is not None: self._render_live(message); self._enrollment_frame(message.frame)
        except queue.Empty: pass
        self.root.after(30, self._poll)

    def _render_live(self, message: FrameMessage) -> None:
        import cv2
        result, metrics = message.result, message.metrics or {}; self.last_frame, self.last_metrics = message.frame, metrics
        counts = {kind: sum(item["kind"] == kind for item in result["detections"]) for kind in ("face", "person", "laptop", "phone")}
        signature = tuple(sorted(counts.items()))
        if signature != self.last_detection_signature:
            self.last_detection_signature = signature
            self._event("info", "Live detections", f"{counts['face']} faces · {counts['person']} people · {counts['laptop']} laptops · {counts['phone']} phones")
        self.count_text.configure(text=f"{counts['face']} faces\n{counts['person']} people\n{counts['laptop']} laptops\n{counts['phone']} phones")
        fps = metrics.get("fps") or 0; latency = metrics.get("end_to_end_ms_avg") or 0
        self.health_text.configure(text="Running", fg=self.GREEN); self.live_status.configure(text=f"Camera running · {fps} FPS", fg=self.GREEN); self.performance_text.configure(text=f"{fps} FPS · {latency} ms end-to-end\nDropped {metrics.get('dropped_frames', 0)} · reconnects {metrics.get('reconnects', 0)}")
        self.diag_summary.configure(text=f"Live test is running. Capture {metrics.get('capture_ms_avg') or '—'} ms · face {metrics.get('face_ms_avg') or '—'} ms · object {metrics.get('object_ms_avg') or '—'} ms.")
        cards = {"fps": f"{fps}", "latency": f"{latency} ms", "dropped": str(metrics.get("dropped_frames", 0)), "errors": str(metrics.get("errors", 0))}
        for key, value in cards.items(): self.metric_cards[key].configure(text=value)
        visible = {kind for kind, enabled in (("face", self.show_faces), ("person", self.show_persons), ("laptop", self.show_laptops), ("phone", self.show_phones)) if enabled}
        image = cv2.cvtColor(annotate(message.frame.copy(), result, visible), cv2.COLOR_BGR2RGB); h, w = image.shape[:2]; scale = min(950 / w, 650 / h, 1); image = cv2.resize(image, (int(w * scale), int(h * scale)))
        photo = self.tk.PhotoImage(data=(f"P6 {image.shape[1]} {image.shape[0]} 255\n".encode() + image.tobytes()), format="PPM"); self.preview.configure(image=photo, text=""); self.preview.image = photo; self._set_badge()

    def begin_enrollment(self) -> None:
        if not self.worker or not self.worker.is_alive(): self.enroll_feedback.configure(text="Start the camera first. Guided enrollment uses the live preview.", fg=self.AMBER); return
        if not self.person_name.get().strip(): self.enroll_feedback.configure(text="Enter the person's display name first.", fg=self.RED); return
        if not self.consent.get(): self.enroll_feedback.configure(text="Confirm informed consent before beginning enrollment.", fg=self.RED); return
        if not self.config.get("scrfd") or not self.config.get("sface") or not Path(self.config["scrfd"]).is_file() or not Path(self.config["sface"]).is_file(): self.enroll_feedback.configure(text="Enrollment needs the optional SCRFD and SFace models. Select them in Set up models.", fg=self.AMBER); return
        self.enrolling = True; self.samples.clear(); self.blurs.clear(); self.enroll_feedback.configure(text="Looking for one sharp face. Keep your face centred and steady; samples are accepted automatically.", fg=self.BLUE); self._event("info", "Guided enrollment started", "Collecting five quality-approved samples in memory only.")

    def cancel_enrollment(self) -> None:
        self.enrolling = False; self.samples.clear(); self.blurs.clear(); self.enroll_feedback.configure(text="Enrollment cancelled. Nothing was saved.", fg=self.MUTED)

    def _enrollment_frame(self, frame: Any) -> None:
        if not self.enrolling: return
        try:
            detector = getattr(self, "_enrollment_detector", None); recognizer = getattr(self, "_enrollment_recognizer", None)
            if detector is None:
                self._enrollment_detector = detector = SCRFD500MDetector(Path(self.config["scrfd"])); self._enrollment_recognizer = recognizer = SFaceRecognizer(Path(self.config["sface"]))
            vector, blur = quality_embedding(frame, detector, recognizer); self.samples.append(vector); self.blurs.append(blur); count = len(self.samples)
            self.enroll_feedback.configure(text=f"Sample {count} of 5 accepted · sharpness {blur:.0f}." + (" Keep steady…" if count < 5 else " Saving your local template…"), fg=self.GREEN)
            if count >= 5: self._save_enrollment()
        except ValueError as exc:
            self.enroll_feedback.configure(text=f"Waiting for a better frame: {exc}", fg=self.AMBER)
        except Exception as exc:
            self.enrolling = False; self.enroll_feedback.configure(text=f"Enrollment stopped: {exc}", fg=self.RED); self._event("error", "Enrollment stopped", str(exc))

    def _save_enrollment(self) -> None:
        try:
            model_hash = sha256_file(Path(self.config["sface"])); embedding = enrolled_embedding(self.samples)
            gallery = parse_gallery(GALLERY_PATH, model_hash) if GALLERY_PATH.exists() else new_gallery(model_hash, int(embedding.size)); person_id = "local-" + uuid.uuid4().hex[:12]
            quality = {"min_face_size_px": 80, "min_blur": 40, "observed_blur_min": round(min(self.blurs), 2), "observed_blur_max": round(max(self.blurs), 2)}
            write_gallery_safely(GALLERY_PATH, update_participant(gallery, person_id, embedding, len(self.samples), quality, self.person_name.get().strip()))
            self.enrolling = False; self.enroll_feedback.configure(text=f"{self.person_name.get().strip()} was saved locally with five quality-approved samples. No images were retained.", fg=self.GREEN); self._event("info", "Person enrolled", "A local template was saved; no face image was retained."); self._refresh_people()
        except Exception as exc:
            self.enrolling = False; self.enroll_feedback.configure(text=f"Could not save enrollment: {exc}", fg=self.RED); self._event("error", "Enrollment save failed", str(exc))

    def _refresh_people(self) -> None:
        if not hasattr(self, "people_list"): return
        self.people_list.delete(0, "end"); self.people: list[dict[str, Any]] = []
        self.remove_person_button.configure(state="disabled")
        try: entries = parse_gallery(GALLERY_PATH)["participants"] if GALLERY_PATH.exists() else []
        except ValueError as exc: self._event("error", "Private gallery unavailable", str(exc)); return
        query = self.people_search.get().lower() if hasattr(self, "people_search") else ""
        for entry in entries:
            if query in entry["display_name"].lower() or query in entry["participant_id"].lower():
                self.people.append(entry); self.people_list.insert("end", f"{entry['display_name']}  ·  {entry['sample_count']} samples  ·  local only")
        if not self.people: self.people_list.insert("end", "No saved people yet. Use the enrollment panel to add someone.")

    def remove_selected(self) -> None:
        from tkinter import messagebox
        index = self.people_list.curselection()
        if not index or not self.people: self._event("warning", "No person selected", "Choose a saved person first."); return
        person = self.people[index[0]]
        if not messagebox.askyesno("Remove local template", f"Remove the local template for {person['display_name']}? A local backup will be kept.", parent=self.root): return
        try:
            gallery = parse_gallery(GALLERY_PATH); write_gallery_safely(GALLERY_PATH, delete_participant(gallery, person["participant_id"])); self._event("info", "Person removed", "The local template was removed and a timestamped backup was kept."); self._refresh_people()
        except Exception as exc: self._event("error", "Could not remove person", str(exc))

    def _ground_truth(self) -> GroundTruth:
        bands = {"Not specified": None, "Near": "near", "Middle": "middle", "Far": "far"}
        return GroundTruth(frozenset(kind for kind, value in self.truth_flags.items() if value.get()), bands[self.distance.get()])

    def prepare_export(self) -> None:
        if self.worker and self.worker.is_alive(): self._event("warning", "Stop the camera first", "Finish the current live test before starting a new saved session."); return
        provenance = {"models": {key: value for key, value in {"yolo_sha256": _model_hash(self.config.get("yolo", "")), "scrfd_sha256": _model_hash(self.config.get("scrfd", ""))}.items() if value}, "distance": self.distance.get(), "privacy": {"identity_publication": "disabled", "face_person_association": "disabled", "captures": "not_retained"}}
        try:
            self.exporter = DiagnosticExport(APP_DIR / "diagnostics", provenance); self.export_status.configure(text=f"Saved session prepared. Return to Live test and start the camera.\n{self.exporter.directory}", fg=self.GREEN); self._event("info", "Saved session prepared", "Start the camera to record JSONL and CSV diagnostics without private identity data.")
        except FileExistsError: self._event("error", "Could not prepare export", "Please try again; a unique output folder could not be allocated.")

    def _event(self, level: str, title: str, detail: str) -> None:
        stamp = time.strftime("%H:%M:%S"); self.events.append((level, f"{stamp}  {title}", detail)); self.events = self.events[-100:]; self._render_events()
        if hasattr(self, "activity_text"):
            recent = self.events[-4:]; self.activity_text.configure(text="\n\n".join(f"{title}\n{detail}" for _level, title, detail in recent))

    def _render_events(self) -> None:
        if not hasattr(self, "event_list"): return
        chosen = self.log_filter.get().lower() if hasattr(self, "log_filter") else "all"; self.event_list.delete(0, "end")
        for level, title, detail in self.events:
            if chosen != "all" and level != chosen[:-1]: continue
            self.event_list.insert("end", f"{level.upper():7}  {title}  —  {detail}")

    def close(self) -> None:
        self.halt(); self.root.destroy()


def main() -> int:
    import tkinter as tk
    root = tk.Tk(); SpectraConsole(root); root.mainloop(); return 0


if __name__ == "__main__": raise SystemExit(main())
