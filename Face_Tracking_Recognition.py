import cv2
import numpy as np
import onnxruntime
import threading
import queue
import time
import faiss
from ultralytics import YOLO
from utils.log_manager import LogManager
from utils.face_utils import (
    align_face,
    preprocess_input,
    non_max_suppression,
    check_frontal_face,
    preprocess_face
)

# ==== CONFIG ====
YOLO_FACE_MODEL = "models/yolov5n-0.5.onnx"
YOLO_PERSON_MODEL = "models/yolo11n.pt"
EMBED_MODEL = "models/webface_r50.onnx"
# ACTIVE_REGION = (550, 600, 1700, 1400)
ACTIVE_REGION = (50,50,1200,650)

# ==== Load Embedding Database ====
embed_session = onnxruntime.InferenceSession(EMBED_MODEL, providers=["CPUExecutionProvider"])
embed_input_name = embed_session.get_inputs()[0].name

database = np.load("embeddings/embeddings.npy").astype(np.float32)
database /= np.linalg.norm(database, axis=1, keepdims=True)
names = np.load("embeddings/names.npy")

index = faiss.IndexFlatL2(database.shape[1])
index.add(database)

# ==== Load Models ====
face_detector = onnxruntime.InferenceSession(YOLO_FACE_MODEL, providers=["CPUExecutionProvider"])
yolo_person = YOLO(YOLO_PERSON_MODEL)
# yolo_person.to("cuda")

# ==== Queues ====
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# ==== Log Manager ====
log_manager = LogManager()


def in_active_region(x1, y1, x2, y2):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    xmin, ymin, xmax, ymax = ACTIVE_REGION
    return xmin <= cx <= xmax and ymin <= cy <= ymax

def capture_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not ret or frame is None:
            continue  # hoặc break tùy logic
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.05)



def process_person(box, frame, draw_frame):
    if box.id is None:
        return

    track_id = int(box.id.item())
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    if not in_active_region(x1, y1, x2, y2):
        return

    roi = frame[y1:y2, x1:x2].copy()

    inp, scale, pad = preprocess_input(roi)
    outs = face_detector.run(None, {face_detector.get_inputs()[0].name: inp})
    dets = non_max_suppression(outs[0], outs[1], outs[2])

    log_manager.log_appear(track_id)
    log_manager.update_last_seen(track_id, person_image=roi)

    for box_face, lm in dets:
        left, top = pad
        scale_inv = 1.0 / scale

        fx1 = int((box_face[0] - left) * scale_inv + x1)
        fy1 = int((box_face[1] - top) * scale_inv + y1)
        fx2 = int((box_face[2] - left) * scale_inv + x1)
        fy2 = int((box_face[3] - top) * scale_inv + y1)

        landmarks = [(int((lm[i] - left) * scale_inv + x1),
                      int((lm[i + 1] - top) * scale_inv + y1)) for i in range(0, 10, 2)]

        aligned_face = align_face(frame, landmarks)
        pose_label, _ = check_frontal_face(lm)

        color = (0, 255, 0) if "Yaw" not in pose_label else (0, 0, 255)
        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.rectangle(draw_frame, (fx1, fy1), (fx2, fy2), color, 2)
        for (lx, ly) in landmarks:
            cv2.circle(draw_frame, (lx, ly), 2, (255, 0, 0), -1)
        cv2.putText(draw_frame, pose_label, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if "Yaw" not in pose_label:
            face_input = preprocess_face(aligned_face)
            face_embed = embed_session.run(None, {embed_input_name: face_input})[0]
            face_embed = face_embed.astype(np.float32)
            face_embed /= np.linalg.norm(face_embed, axis=1, keepdims=True)

            D, I = index.search(face_embed, k=1)
            distance = D[0][0]
            idx = I[0][0]
            confidence = max(0.0, 1.0 - distance / 2.0)
            name = names[idx] if confidence > 0.3 else "Unknown"
            text = f"{name} ({confidence * 100:.1f}%)"
            cv2.putText(draw_frame, text, (fx1, fy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            log_manager.log_identified(track_id, name, confidence, pose_label, aligned_face, roi)

def process_thread():
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        draw_frame = frame.copy()

        results = yolo_person.track(source=frame, persist=True, classes=[0], verbose=False)
        boxes = results[0].boxes

        threads = []
        for box in boxes:
            t = threading.Thread(target=process_person, args=(box, frame, draw_frame))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        xmin, ymin, xmax, ymax = ACTIVE_REGION
        cv2.rectangle(draw_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(draw_frame, "Active Region", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        log_manager.cleanup_tracks(timeout=3)

        if not result_queue.full():
            result_queue.put(draw_frame)

# ==== START ====
url1 = "rtsp://admin:Hicas%402024%21@10.0.10.120:554"
url2 = "rtsp://10.0.10.53:8554/live/7c158392-772c-4497-8b62-000e9861ad6b"
url3 = "rtsp://10.0.10.55:8554/live/c64f766f-b2d6-4bf6-a199-1c48ae55e341"
cap = cv2.VideoCapture(0) #url3, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 810)

threading.Thread(target=capture_thread, args=(cap,), daemon=True).start()
threading.Thread(target=process_thread, daemon=True).start()

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 1440, 810)

while True:
    if result_queue.empty():
        time.sleep(0.01)
        continue
    display_frame = result_queue.get()
    cv2.imshow("Tracking", display_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
