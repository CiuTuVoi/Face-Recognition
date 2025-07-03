import os
import csv
import cv2
import datetime
from collections import defaultdict

def current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

class LogManager:
    def __init__(self, log_dir="logs", face_dir="logs/faces", person_dir="logs/persons", min_confidence=0.3):
        self.log_dir = log_dir
        self.face_dir = face_dir
        self.person_dir = person_dir
        self.min_confidence = min_confidence

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.face_dir, exist_ok=True)
        os.makedirs(self.person_dir, exist_ok=True)

        # Trạng thái theo track_id
        self.track_states = defaultdict(lambda: {
            "first_seen": None,
            "last_seen": None,
            "identified": False,
            "name": "Unknown",
            "face_image": None,
            "person_image": None,
            "confidence": 0
        })

        self.log_file_path = os.path.join(self.log_dir, f"log_{current_date()}.csv")
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "track_id", "action", "name", "confidence", "pose", "face_image", "person_image"])

    def is_blurry(self, image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return var < threshold

    def log_appear(self, track_id):
        now = current_timestamp()
        state = self.track_states[track_id]
        if state["first_seen"] is None:
            state["first_seen"] = now
            self._write_log(now, track_id, "appear", "Unknown", "", "", "", "")

    def log_disappear(self, track_id):
        now = current_timestamp()
        state = self.track_states.get(track_id)
        if state and not state.get("disappeared", False):
            state["disappeared"] = True
            face_path, person_path = "", ""

            # Ghi ảnh nếu chưa từng nhận diện
            if not state["identified"]:
                if state["face_image"] is not None:
                    face_path = self._save_image(state["face_image"], self.face_dir, track_id)
                if state["person_image"] is not None:
                    person_path = self._save_image(state["person_image"], self.person_dir, track_id)

            self._write_log(now, track_id, "disappear", state["name"], state["confidence"], "", face_path, person_path)

    def log_identified(self, track_id, name, confidence, pose_label, face_image, person_image):
        now = current_timestamp()
        state = self.track_states[track_id]
        if state["identified"]:
            return

        if confidence < self.min_confidence or self.is_blurry(face_image):
            return

        face_path = self._save_image(face_image, self.face_dir, track_id)
        person_path = self._save_image(person_image, self.person_dir, track_id)

        state["identified"] = True
        state["name"] = name
        state["confidence"] = confidence
        state["face_image"] = face_image
        state["person_image"] = person_image

        self._write_log(now, track_id, "identified", name, f"{confidence:.2f}", pose_label, face_path, person_path)

    def update_last_seen(self, track_id, face_image=None, person_image=None):
        state = self.track_states[track_id]
        state["last_seen"] = datetime.datetime.now()
        if face_image is not None:
            state["face_image"] = face_image
        if person_image is not None:
            state["person_image"] = person_image

    def cleanup_tracks(self, timeout=3):
        now = datetime.datetime.now()
        to_remove = []
        for track_id, state in self.track_states.items():
            last_seen = state["last_seen"]
            if last_seen and (now - last_seen).total_seconds() > timeout:
                self.log_disappear(track_id)
                to_remove.append(track_id)
        for tid in to_remove:
            del self.track_states[tid]

    def _save_image(self, img, folder, track_id):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"track_{track_id}_{now}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, img)
        return path

    def _write_log(self, timestamp, track_id, action, name, confidence, pose, face_img_path, person_img_path):
        with open(self.log_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, track_id, action, name, confidence, pose, face_img_path, person_img_path])
