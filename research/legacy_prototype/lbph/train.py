"""Archived LBPH training utility. Outputs only to ignored local_data/."""
import json

import cv2
import numpy as np

from . import config


def train_model():
    if not config.DATASET_PATH.exists() or not any(config.DATASET_PATH.iterdir()):
        print("Dataset is empty. Enroll at least one person first (enroll.py).")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, label_map = [], [], {}
    next_id = 0
    for person_dir in sorted(config.DATASET_PATH.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith("."):
            continue
        image_files = [path for path in person_dir.iterdir() if not path.name.startswith(".")]
        if not image_files:
            print(f"Skipping '{person_dir.name}' - no images found.")
            continue
        label_map[next_id] = person_dir.name
        for path in image_files:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(cv2.resize(image, config.FACE_SIZE))
                labels.append(next_id)
        next_id += 1
    if not faces:
        print("No valid face images found in dataset/. Nothing to train.")
        return
    config.LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    recognizer.train(faces, np.array(labels))
    recognizer.save(str(config.MODEL_PATH))
    config.LABELS_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"Model trained on {len(faces)} images across {len(label_map)} people.")


if __name__ == "__main__":
    train_model()
