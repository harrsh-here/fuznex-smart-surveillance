"""Consent-based, local-only SFace enrollment.  No face images are retained."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ai.detection import SCRFD500MDetector
from ai.local_gallery import (delete_participant, new_gallery,
                              parse_gallery, sha256_file, update_participant,
                              write_gallery_safely)
from ai.recognition import SFaceRecognizer, enrolled_embedding

CONSENT_PHRASE = "I_CONSENT"


def quality_embedding(frame: np.ndarray, detector: Any, recognizer: Any,
                      min_face_size_px: int = 80, min_blur: float = 40.0) -> tuple[np.ndarray, float]:
    """Return one quality-approved embedding, never saving the supplied frame."""
    faces = detector.detect(frame)
    if len(faces) != 1:
        raise ValueError("exactly one face must be visible for each enrollment sample")
    box, landmarks = faces[0]
    if min(float(box[2] - box[0]), float(box[3] - box[1])) < min_face_size_px:
        raise ValueError("face is below the enrollment size gate")
    embedding, crop = recognizer.embed(frame, box, landmarks)
    import cv2
    blur = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    if blur < min_blur:
        raise ValueError("face crop is below the sharpness gate")
    return embedding, blur


def collect_image_samples(paths: list[Path], detector: Any, recognizer: Any,
                          min_face_size_px: int, min_blur: float) -> tuple[list[np.ndarray], list[float]]:
    import cv2
    vectors: list[np.ndarray] = []; blurs: list[float] = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("an enrollment image could not be read")
        vector, blur = quality_embedding(frame, detector, recognizer, min_face_size_px, min_blur)
        vectors.append(vector); blurs.append(blur)
    return vectors, blurs


def _camera_samples(camera: str, count: int, detector: Any, recognizer: Any,
                    min_face_size_px: int, min_blur: float) -> tuple[list[np.ndarray], list[float]]:
    import cv2
    source = int(camera) if camera.isdecimal() else camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError("camera could not be opened")
    vectors: list[np.ndarray] = []; blurs: list[float] = []
    try:
        print("Enrollment captures are processed in memory only. SPACE accepts a sample; q cancels.")
        while len(vectors) < count:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("camera frame could not be read")
            preview = frame.copy(); message = f"Approved: {len(vectors)}/{count}"
            try:
                vector, blur = quality_embedding(frame, detector, recognizer, min_face_size_px, min_blur)
                message += f" · ready (sharpness {blur:.0f})"
            except ValueError as exc:
                vector = None; blur = 0.0; message += f" · {exc}"
            cv2.putText(preview, message, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
            cv2.imshow("FuzNex private SFace enrollment", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise ValueError("enrollment cancelled; gallery was not changed")
            if key == ord(" ") and vector is not None:
                vectors.append(vector); blurs.append(blur)
    finally:
        capture.release(); cv2.destroyAllWindows()
    return vectors, blurs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True, help="ignored local JSON gallery path")
    parser.add_argument("--sface", type=Path)
    parser.add_argument("--scrfd", type=Path)
    parser.add_argument("--participant-id", required=True, help="local generated ID (any non-empty value)")
    parser.add_argument("--consent", help=f"must be exactly {CONSENT_PHRASE}")
    parser.add_argument("--image", type=Path, action="append", default=[], help="fresh consenting sample; repeat at least 3 times")
    parser.add_argument("--camera", help="camera/IP source for interactive, in-memory capture")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--min-face-size", type=int, default=80)
    parser.add_argument("--min-blur", type=float, default=40.0)
    parser.add_argument("--create", action="store_true", help="create a new empty private gallery")
    parser.add_argument("--delete", action="store_true", help="delete only this participant's local template")
    parser.add_argument("--confirm-delete", help="repeat the participant ID to confirm deletion")
    args = parser.parse_args()
    if args.delete:
        if args.confirm_delete != args.participant_id:
            parser.error("--delete requires --confirm-delete with the same participant ID")
        gallery = parse_gallery(args.gallery, sha256_file(args.sface) if args.sface else None)
        backup = write_gallery_safely(args.gallery, delete_participant(gallery, args.participant_id))
        print(f"Deleted local template for {args.participant_id}. Backup created: {bool(backup)}")
        return 0
    if args.consent != CONSENT_PHRASE:
        parser.error(f"enrollment requires --consent {CONSENT_PHRASE}")
    if not args.sface or not args.scrfd:
        parser.error("enrollment requires both --scrfd and --sface")
    if bool(args.image) == bool(args.camera):
        parser.error("provide either repeated --image samples or --camera, not both")
    if args.samples < 3 or args.min_face_size <= 0 or args.min_blur < 0:
        parser.error("samples must be at least 3; quality gates must be non-negative")
    model_hash = sha256_file(args.sface)
    detector = SCRFD500MDetector(args.scrfd)
    recognizer = SFaceRecognizer(args.sface)
    if args.image:
        vectors, blurs = collect_image_samples(args.image, detector, recognizer, args.min_face_size, args.min_blur)
    else:
        vectors, blurs = _camera_samples(args.camera, args.samples, detector, recognizer, args.min_face_size, args.min_blur)
    if len(vectors) < 3:
        parser.error("at least three quality-approved fresh samples are required")
    embedding = enrolled_embedding(vectors)
    if args.create and args.gallery.exists():
        parser.error("--create refuses to replace an existing gallery; use an update without --create")
    gallery = new_gallery(model_hash, int(embedding.size)) if args.create else parse_gallery(args.gallery, model_hash)
    quality = {"min_face_size_px": args.min_face_size, "min_blur": args.min_blur,
               "observed_blur_min": round(min(blurs), 3), "observed_blur_max": round(max(blurs), 3)}
    backup = write_gallery_safely(args.gallery, update_participant(gallery, args.participant_id, embedding, len(vectors), quality))
    print(f"Locally enrolled {args.participant_id} from {len(vectors)} fresh approved samples. Backup created: {bool(backup)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
