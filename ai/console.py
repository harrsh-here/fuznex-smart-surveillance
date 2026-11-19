"""SPECTRA DIAG local viewer and headless laptop replay.

``python -m ai.console --yolo C:\\models\\yolo11n.pt --camera 0``
``python -m ai.console --yolo C:\\models\\yolo11n.pt --image C:\\test\\laptop.jpg``
"""
from __future__ import annotations
import argparse
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ai.combined import CombinedPipeline
from ai.object_detection import YoloConfig, YoloPersonLaptopDetector
from ai.spectra_diag import DiagnosticExport, RunMetrics, replay_frames


class EmptyFacePipeline:
    """Object diagnosis does not load or expose identity functionality."""
    def process_frame(self, _frame: Any) -> Any:
        return type("FaceFrame", (), {"faces": ()})()


def annotate(frame: Any, result: dict[str, Any], visible: set[str]) -> Any:
    import cv2
    colors={"face":(255,182,72), "person":(91,213,164), "laptop":(101,163,255)}
    for item in result["detections"]:
        if item["kind"] not in visible: continue
        x1,y1,x2,y2=(int(v) for v in item["bbox_xyxy"]); color=colors[item["kind"]]
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{item['kind'].upper()} {item['detection_confidence']:.0%}",(x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    return frame


class CameraWorker(threading.Thread):
    """Bounded hand-off makes render backlog and reconnects measurable."""
    def __init__(self,pipeline:CombinedPipeline,source:str,messages:queue.Queue,stop:threading.Event,export:DiagnosticExport)->None:
        super().__init__(daemon=True); self.pipeline=pipeline; self.source=source; self.messages=messages; self.stop=stop; self.export=export; self.metrics=RunMetrics()
    def offer(self,item:dict[str,Any])->None:
        try:self.messages.put_nowait(item)
        except queue.Full:self.metrics.dropped_frames+=1
    def run(self)->None:
        import cv2
        source=int(self.source) if self.source.isdecimal() else self.source; cap=None; epoch=uuid.uuid4().hex; seq=0
        while not self.stop.is_set():
            if cap is None or not cap.isOpened():
                cap=cv2.VideoCapture(source)
                if not cap.isOpened():
                    self.metrics.errors+=1; self.offer({"status":"Camera/IP stream unavailable; retrying…"}); time.sleep(1); continue
                if seq:self.metrics.reconnects+=1
                self.offer({"status":"Source connected"})
            acquired=time.perf_counter(); ok,frame=cap.read(); self.metrics.capture_ms.append((time.perf_counter()-acquired)*1000)
            if not ok:
                self.metrics.dropped_frames+=1; cap.release(); cap=None; self.offer({"status":"Frame lost; reconnecting…"}); continue
            result=self.pipeline.process_frame(frame,f"source:{self.source}",epoch,seq,acquired_at=acquired); seq+=1
            self.metrics.frames+=1; self.metrics.inference_ms.append(result["processing"]["total_ms"]); self.metrics.end_to_end_ms.append(result["processing"]["capture_to_output_ms"]); self.metrics.errors+=len(result["processing_errors"]); self.export.record(result,self.metrics)
            self.offer({"frame":frame,"result":result,"metrics":self.metrics.snapshot()})
        if cap is not None:cap.release()
        self.export.close()


def launch_camera(pipeline:CombinedPipeline,source:str="0",output_dir:str="diagnostics")->None:
    """Windows UI with independent overlays and end-to-end pipeline health."""
    import cv2
    import tkinter as tk
    from tkinter import ttk
    root=tk.Tk(); root.title("FuzNex · SPECTRA DIAG"); root.geometry("1280x820"); root.minsize(1000,650); root.configure(bg="#101826")
    style=ttk.Style(root); style.theme_use("clam"); style.configure("TFrame",background="#101826"); style.configure("Card.TFrame",background="#182335"); style.configure("TLabel",background="#182335",foreground="#e6edf7",font=("Segoe UI",10)); style.configure("Title.TLabel",background="#101826",foreground="#f5f8ff",font=("Segoe UI Semibold",22)); style.configure("Sub.TLabel",background="#101826",foreground="#91a4c2",font=("Segoe UI",10)); style.configure("TButton",font=("Segoe UI Semibold",10),padding=(12,7))
    visible={"face","person","laptop"}; messages:queue.Queue=queue.Queue(maxsize=2); stop=threading.Event(); worker:CameraWorker|None=None; source_var=tk.StringVar(value=source); status=tk.StringVar(value="Ready · identity publication and face-to-person association are disabled"); perf=tk.StringVar(value="Awaiting source"); counts={k:tk.StringVar(value="0") for k in visible}
    top=ttk.Frame(root); top.pack(fill="x",padx=26,pady=(22,8)); ttk.Label(top,text="SPECTRA DIAG",style="Title.TLabel").pack(anchor="w"); ttk.Label(top,text="Capture → inference → render diagnostics · local-only · no candidate names",style="Sub.TLabel").pack(anchor="w")
    controls=ttk.Frame(root,style="Card.TFrame"); controls.pack(fill="x",padx=26,pady=10); ttk.Label(controls,text="Camera / IP stream").pack(side="left",padx=(16,6),pady=14); ttk.Entry(controls,textvariable=source_var,width=44).pack(side="left",pady=14)
    def toggle(k:str,v:tk.BooleanVar)->None: visible.add(k) if v.get() else visible.discard(k)
    for k in ("face","person","laptop"):
        v=tk.BooleanVar(value=True); ttk.Checkbutton(controls,text=k.title(),variable=v,command=lambda k=k,v=v:toggle(k,v)).pack(side="left",padx=8)
    body=ttk.Frame(root); body.pack(fill="both",expand=True,padx=26,pady=10); preview=ttk.Label(body,text="Start a camera or IP stream to inspect the live render",anchor="center"); preview.pack(side="left",fill="both",expand=True); side=ttk.Frame(body,style="Card.TFrame",width=260); side.pack(side="right",fill="y",padx=(14,0)); side.pack_propagate(False)
    ttk.Label(side,text="LIVE COUNTS",font=("Segoe UI Semibold",11)).pack(anchor="w",padx=18,pady=(20,8))
    for k in ("face","person","laptop"): ttk.Label(side,textvariable=counts[k],font=("Segoe UI Semibold",25)).pack(anchor="w",padx=18); ttk.Label(side,text=k.title()).pack(anchor="w",padx=20,pady=(0,8))
    ttk.Separator(side).pack(fill="x",padx=18,pady=9); ttk.Label(side,text="PIPELINE HEALTH",font=("Segoe UI Semibold",11)).pack(anchor="w",padx=18); ttk.Label(side,textvariable=perf,wraplength=220,justify="left").pack(anchor="w",padx=18,pady=8)
    bottom=ttk.Frame(root); bottom.pack(fill="x",padx=26,pady=(4,24)); ttk.Label(bottom,textvariable=status,style="Sub.TLabel").pack(side="left")
    def start()->None:
        nonlocal worker
        if worker and worker.is_alive():return
        stop.clear(); export=DiagnosticExport(output_dir,{"object":getattr(pipeline.object_detector,"provenance",lambda:{"status":"not_available"})(),"privacy":{"identity_publication":"disabled","face_person_association":"disabled"}}); worker=CameraWorker(pipeline,source_var.get().strip(),messages,stop,export); worker.start(); status.set(f"Running · trace: {export.directory}")
    def halt()->None:stop.set(); status.set("Stopping capture…")
    ttk.Button(bottom,text="Start diagnostic",command=start).pack(side="right",padx=5); ttk.Button(bottom,text="Stop",command=halt).pack(side="right",padx=5)
    def poll()->None:
        try:
            while True:
                item=messages.get_nowait()
                if "status" in item:status.set(item["status"]); continue
                result=item["result"]
                for k in counts:counts[k].set(str(sum(d["kind"]==k for d in result["detections"])))
                m=item["metrics"]; perf.set(f"{m['fps'] or '—'} FPS\n{m['end_to_end_ms_avg'] or '—'} ms end-to-end\nDropped {m['dropped_frames']} · errors {m['errors']}\nReconnects {m['reconnects']}")
                rgb=cv2.cvtColor(annotate(item["frame"].copy(),result,visible),cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; s=min(900/w,640/h,1); rgb=cv2.resize(rgb,(int(w*s),int(h*s))); photo=tk.PhotoImage(data=(f"P6 {rgb.shape[1]} {rgb.shape[0]} 255\n".encode()+rgb.tobytes()),format="PPM"); preview.configure(image=photo,text=""); preview.image=photo
        except queue.Empty:pass
        root.after(30,poll)
    root.protocol("WM_DELETE_WINDOW",lambda:(halt(),root.destroy())); root.after(30,poll); root.mainloop()


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--yolo",required=True,type=Path); parser.add_argument("--camera",default="0"); parser.add_argument("--image",type=Path); parser.add_argument("--input-size",type=int,default=640); parser.add_argument("--confidence",type=float,default=.25); parser.add_argument("--output-dir",default="diagnostics"); args=parser.parse_args()
    detector=YoloPersonLaptopDetector(YoloConfig(args.yolo,args.input_size,args.confidence)); pipeline=CombinedPipeline(EmptyFacePipeline(),detector)
    if args.image:
        import cv2
        image=cv2.imread(str(args.image))
        if image is None:raise SystemExit(f"Cannot read image: {args.image}")
        export=DiagnosticExport(args.output_dir,{"object":detector.provenance(),"mode":"offline-replay"}); results,metrics=replay_frames(pipeline,[image],f"image:{args.image.name}",exporter=export); export.close(); print({"detections":results[0]["detections"],"metrics":metrics.snapshot(),"export":str(export.directory)}); return 0
    launch_camera(pipeline,args.camera,args.output_dir); return 0

if __name__=="__main__":raise SystemExit(main())
