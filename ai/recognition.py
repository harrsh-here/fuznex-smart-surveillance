"""SFace alignment, embedding and experimental identity decisions."""
from __future__ import annotations
from pathlib import Path
import numpy as np

SFACE_REFERENCE_THRESHOLD=0.363
SFACE_REFERENCE_SOURCE="OpenCV SFace LFW cosine reference"

def cosine(a: np.ndarray,b: np.ndarray) -> float: return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
class SFaceRecognizer:
    def __init__(self, model_path: str | Path):
        import cv2
        self.cv2=cv2; self.model=cv2.FaceRecognizerSF.create(str(model_path),"")
    def embed(self, frame: np.ndarray, box: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        row=np.r_[box[0],box[1],box[2]-box[0],box[3]-box[1],landmarks.reshape(-1),box[4]].astype("float32")
        crop=self.model.alignCrop(frame,row); vector=self.model.feature(crop).reshape(-1); return vector/np.linalg.norm(vector),crop

def decide(embedding: np.ndarray, templates: dict[str,np.ndarray], threshold: float=SFACE_REFERENCE_THRESHOLD, band: float=.05) -> tuple[str,str|None,float|None]:
    if not templates: return "unknown",None,None
    identity,score=max(((name,cosine(embedding,v)) for name,v in templates.items()), key=lambda item:item[1])
    state="known" if score>=threshold else "uncertain" if score>=threshold-band else "unknown"
    # A nearest template is not an identity assertion unless the configured
    # experimental rule actually classifies the face as known.
    return state, identity if state=="known" else None, score

def enrolled_embedding(samples: list[np.ndarray]) -> np.ndarray:
    if len(samples)<3: raise ValueError("At least three quality-approved embeddings are required.")
    value=np.mean(samples,axis=0); return value/np.linalg.norm(value)
