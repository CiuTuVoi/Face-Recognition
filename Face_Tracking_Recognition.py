import cv2
import numpy as np
import onnxruntime
import faiss
import threading
import queue
import time
import os
import csv
from ultralytics import YOLO
from utils.face_utils import align_face, preprocess_face, preprocess_input, non_max_suppression

YOLO_FACE_MODEL = "models/yolov5n-0.5.onnx"
YOLO_PERSON_MODEL = "models/yolov8n.pt"
EMBED_MODEL = "models/webface_r50.onnx"
EMB_PATH = "embeddings/embeddings.npy"
NAME_PATH = "embeddings/names.npy"
LOG_DIR = "logs"
SNAP_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "log.csv")
SIM_THRESHOLD = 0.6
ACTIVE_REGION = (400, 530, 2000, 1350)  # (x_min, y_min, x_max, y_max)
ACTIVE_REGION = (50, 50, 1200, 650)  # (x_min, y_min, x_max, y_max)

os.makedirs(SNAP_DIR, exist_ok=True)

face_detector = onnxruntime.InferenceSession(YOLO_FACE_MODEL, providers=["CUDAExecutionProvider"])
embed_model = onnxruntime.InferenceSession(EMBED_MODEL, providers=["CUDAExecutionProvider"])
embed_input = embed_model.get_inputs()[0].name
embed_output = embed_model.get_outputs()[0].name

embs = np.load(EMB_PATH).astype('float32')
names = list(np.load(NAME_PATH))
faiss.normalize_L2(embs)
id_map = {i: i for i in range(len(names))}
next_id = max(id_map.values(), default=-1) + 1

index = faiss.IndexIDMap(faiss.IndexFlatIP(embs.shape[1]))
index.add_with_ids(embs, np.array(list(id_map.values()), dtype=np.int64))

yolo_person = YOLO(YOLO_PERSON_MODEL)
person_faces = {}
track_status = {}

frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

def in_active_region(x1, y1, x2, y2):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    xmin, ymin, xmax, ymax = ACTIVE_REGION
    return xmin <= cx <= xmax and ymin <= cy <= ymax

def log_event_summary(track_id, name, sim, t_start, t_end, best_img_path):
    row = [t_start, t_end, track_id, name, f"{sim:.2f}" if sim else "", best_img_path]
    with open(LOG_FILE, "a", newline='') as f:
        csv.writer(f).writerow(row)

def register_new(emb):
    global embs, names, index, id_map, next_id
    new_id = next_id
    next_id += 1
    new_name = f"unknown_{new_id:03d}"
    names.append(new_name)
    emb = emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(emb)
    index.add_with_ids(emb, np.array([new_id], dtype=np.int64))
    id_map[len(names)-1] = new_id
    embs = np.vstack([embs, emb])
    np.save(EMB_PATH, embs)
    np.save(NAME_PATH, np.array(names))
    return new_name, new_id

def recognize_face(frame, box, lm):
    aligned = align_face(frame, lm)
    inp_face = preprocess_face(aligned)
    emb = embed_model.run([embed_output], {embed_input: inp_face})[0][0]
    emb = emb / np.linalg.norm(emb)
    query = emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query)
    D, I = index.search(query, k=1)
    sim = float(D[0][0])
    matched_id = int(I[0][0])
    name = names[list(id_map.values()).index(matched_id)] if sim > SIM_THRESHOLD else None
    if name is None:
        name, matched_id = register_new(emb)
    return name, sim, emb

def capture_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)

def process_thread():
    frame_count = 0
    track_buffer = {}
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        frame = frame_queue.get()
        frame_count += 1
        draw_frame = frame.copy()

        results = yolo_person.track(source=frame, persist=True, classes=[0], verbose=False)
        boxes = results[0].boxes
        current_ids = set()

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            if not in_active_region(x1, y1, x2, y2):
                continue

            track_id = int(box.id[0]) if box.id is not None else i
            current_ids.add(track_id)
            roi = frame[y1:y2, x1:x2].copy()

            if track_id not in track_buffer:
                track_buffer[track_id] = {
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": None,
                    "name": "Unknown",
                    "sim": 0,
                    "best_score": 0,
                    "best_img": None
                }

            if frame_count % 5 == 0:
                inp, scale, pad = preprocess_input(roi)
                outs = face_detector.run(None, {face_detector.get_inputs()[0].name: inp})
                dets = non_max_suppression(outs[0], outs[1], outs[2])
                if len(dets) > 0:
                    box_face, lm = dets[0]
                    left, top = pad
                    landmarks = [(int((lm[i] - left) / scale), int((lm[i + 1] - top) / scale)) for i in range(0, 10, 2)]
                    name, sim, emb = recognize_face(roi, box_face, landmarks)
                    if name != "Unknown" and sim > track_buffer[track_id]["sim"]:
                        track_buffer[track_id]["name"] = name
                        track_buffer[track_id]["sim"] = sim
                        track_buffer[track_id]["best_img"] = roi.copy()
                    elif name == "Unknown" and dets[0][1][-1] > track_buffer[track_id]["best_score"]:
                        track_buffer[track_id]["best_score"] = dets[0][1][-1]
                        track_buffer[track_id]["best_img"] = roi.copy()

            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(draw_frame, f"{track_buffer[track_id]['name']}_{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        to_remove = []
        for tid in list(track_buffer.keys()):
            if tid not in current_ids:
                if tid not in track_status:
                    track_status[tid] = frame_count
                elif frame_count - track_status[tid] > 30:
                    t = track_buffer[tid]
                    t["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    img_path = ""
                    if t["best_img"] is not None:
                        img_path = os.path.join(SNAP_DIR, f"person_{tid}_{int(time.time())}.jpg")
                        cv2.imwrite(img_path, t["best_img"])
                    log_event_summary(tid, t["name"], t["sim"], t["start_time"], t["end_time"], img_path)
                    to_remove.append(tid)
        for tid in to_remove:
            del track_buffer[tid]
            if tid in track_status:
                del track_status[tid]

        xmin, ymin, xmax, ymax = ACTIVE_REGION
        cv2.rectangle(draw_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(draw_frame, "Active Region", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if not result_queue.full():
            result_queue.put(draw_frame)

# ==== START ====
url = "rtsp://admin:Hicas@2024!@10.0.10.120:554"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 810)

t1 = threading.Thread(target=capture_thread, args=(cap,), daemon=True)
t2 = threading.Thread(target=process_thread, daemon=True)
t1.start()
t2.start()

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
