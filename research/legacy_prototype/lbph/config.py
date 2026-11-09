"""Repository-local configuration for the archived LBPH prototype."""
from pathlib import Path

CAMERA_SOURCE = 0
LOCAL_DATA = Path(__file__).resolve().parents[3] / "local_data" / "lbph"
DATASET_PATH = LOCAL_DATA / "dataset"
MODEL_PATH = LOCAL_DATA / "model.yml"
LABELS_PATH = LOCAL_DATA / "labels.json"
ALERTS_PATH = LOCAL_DATA / "cctv_alerts"
HAAR_CASCADE_PATH = LOCAL_DATA / "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
SAMPLES_PER_PERSON = 40
CONFIDENCE_THRESHOLD = 85
UNKNOWN_STREAK_TO_ALERT = 3
SCALE_DOWN_FACTOR = 0.5
FRAME_SKIP = 2
MOTION_MIN_AREA = 300
ALERT_COOLDOWN = 5
HAAR_SCALE_FACTOR = 1.2
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (40, 40)
