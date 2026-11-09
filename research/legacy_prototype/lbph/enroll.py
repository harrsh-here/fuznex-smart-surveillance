"""Archived manual enrollment utility. Images are written only to ignored local_data/."""
import cv2

from . import config

POSE_PROMPTS = [
    "Look straight at the camera", "Look straight at the camera",
    "Turn head slightly LEFT", "Turn head slightly LEFT",
    "Turn head slightly RIGHT", "Turn head slightly RIGHT",
    "Tilt chin down slightly", "Tilt chin up slightly",
    "Look straight - neutral face", "Look straight - smiling",
]


def get_pose_prompt(count, total, wearing_note):
    prompt = POSE_PROMPTS[count % len(POSE_PROMPTS)]
    return f"{prompt}  (WEARING: {wearing_note})" if wearing_note and count >= total // 2 else prompt


def enroll_person(name):
    person_dir = config.DATASET_PATH / name
    person_dir.mkdir(parents=True, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(str(config.HAAR_CASCADE_PATH))
    if face_cascade.empty():
        print(f"ERROR: could not load {config.HAAR_CASCADE_PATH}")
        return
    camera = cv2.VideoCapture(config.CAMERA_SOURCE)
    print(f"Enrolling '{name}'. Press SPACE to capture and q to stop.")
    wearing_note = input("Glasses or cap (optional): ").strip().lower()
    count = 0
    while count < config.SAMPLES_PER_PERSON:
        ok, frame = camera.read()
        if not ok:
            print("Camera read failed.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        roi = None
        if len(faces):
            x, y, width, height = faces[0]
            roi = gray[y:y + height, x:x + width]
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f"[{count}/{config.SAMPLES_PER_PERSON}] {get_pose_prompt(count, config.SAMPLES_PER_PERSON, wearing_note)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE = capture   q = quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Enrolling - SPACE to capture, q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and roi is not None:
            count += 1
            cv2.imwrite(str(person_dir / f"{count}.jpg"), cv2.resize(roi, config.FACE_SIZE))
            print(f"Captured {count}/{config.SAMPLES_PER_PERSON}")
    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {person_dir}")


if __name__ == "__main__":
    person_name = input("Enter the person's name (no spaces preferred): ").strip()
    if person_name:
        enroll_person(person_name)
    else:
        print("Name cannot be empty.")
