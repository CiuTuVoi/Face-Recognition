# face_tracking_faiss.py
import cv2
import threading
import queue
import time
import onnxruntime
import numpy as np
import faiss
import os
import csv
from utils.deep_sort_realtime_custom.deepsort_tracker import DeepSort
from utils.face_utils import align_face, preprocess_face, preprocess_input, non_max_suppression

# ========== Config ==========
YOLO_MODEL = "models/yolov5n-0.5.onnx"
EMBED_MODEL = "models/webface_r50.onnx"
EMB_PATH = "embeddings/embeddings.npy"
NAME_PATH = "embeddings/names.npy"
LOG_DIR = "logs"
SNAP_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "log.csv")
# ROI = (500, 300, 1000, 700) # url
ROI = (60, 60, 500, 400) # 0
SIM_THRESHOLD = 0.5

ios = lambda path: os.makedirs(path, exist_ok=True)
ios(LOG_DIR)
ios(SNAP_DIR)

# ========== Load Models ==========
yolo_session = onnxruntime.InferenceSession(YOLO_MODEL, providers=["CUDAExecutionProvider"])
embed_session = onnxruntime.InferenceSession(EMBED_MODEL, providers=["CUDAExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name
embed_input_name = embed_session.get_inputs()[0].name

# ========== Load FAISS ==========
embs = np.load(EMB_PATH).astype('float32') if os.path.exists(EMB_PATH) else np.empty((0, 512), dtype='float32')
names = list(np.load(NAME_PATH)) if os.path.exists(NAME_PATH) else []
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(512)
if len(embs) > 0:
    index.add(embs)

# ========== Tracker & State ==========
tracker = DeepSort(embedder=None, embedder_fn=lambda x: x, max_age=30)
track_info = {}  # track_id: {name, appear_count, last_seen_frame}
disappear_timeout = 90
frame_count = 0

# ========== Queues ==========
yolo_queue = queue.Queue(maxsize=5)
embedding_queue = queue.Queue(maxsize=5)
tracking_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()

# ========== Utility ==========
def extract_embedding(face_input):
    emb = embed_session.run(None, {embed_input_name: face_input})[0][0]
    return emb / np.linalg.norm(emb)

def in_roi(box, roi):
    x, y, w, h = box
    rx, ry, rw, rh = roi
    return (x >= rx and y >= ry and x + w <= rx + rw and y + h <= ry + rh)

def log_event(track_id, name, event, img=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, track_id, name, event]
    if img is not None:
        fname = f"{name}_{int(time.time())}.jpg"
        path = os.path.join(SNAP_DIR, fname)
        cv2.imwrite(path, img)
        row.append(path)
    with open(LOG_FILE, "a", newline='') as f:
        csv.writer(f).writerow(row)

def register_new_face(emb, img):
    global embs, names, index
    new_name = f"ID_{len(names)+1:03d}"
    names.append(new_name)
    embs = np.vstack([embs, emb])
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(512)
    index.add(embs)
    np.save(EMB_PATH, embs)
    np.save(NAME_PATH, np.array(names))
    log_event(new_name, new_name, "new_registered", img)
    return new_name

# ========== Threads ==========
def thread_camera_detect():
    url = "rtsp://admin:Hicas@2024!@10.0.10.120:554"
    url1 = "rtsp://10.0.10.53:8554/live/7c158392-772c-4497-8b62-000e9861ad6b"
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    global frame_count
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        inp, scale, pad = preprocess_input(frame)
        boxes, scores, landmarks = yolo_session.run(None, {yolo_input_name: inp})
        detections = non_max_suppression(boxes, scores, landmarks)
        yolo_queue.put((frame, detections, scale, pad, frame_count))
    cap.release()

def thread_embedding():
    while not stop_event.is_set():
        try:
            frame, dets, scale, pad, f_id = yolo_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        aligned = []
        for box, lmk in dets:
            x1, y1, x2, y2 = box
            x1 = int((x1 - pad[0]) / scale)
            y1 = int((y1 - pad[1]) / scale)
            x2 = int((x2 - pad[0]) / scale)
            y2 = int((y2 - pad[1]) / scale)
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            if not in_roi(bbox, ROI):
                continue
            lmk = lmk.reshape((5, 2))
            lmk[:, 0] = (lmk[:, 0] - pad[0]) / scale
            lmk[:, 1] = (lmk[:, 1] - pad[1]) / scale
            try:
                aligned_face = align_face(frame, lmk)
                face_input = preprocess_face(aligned_face)
                aligned.append((face_input, frame[y1:y2, x1:x2].copy(), bbox))
            except:
                continue
        embedding_queue.put((frame, aligned, f_id))

def thread_tracking():
    while not stop_event.is_set():
        try:
            frame, faces, f_id = embedding_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        dets, embeds, snapshots, bboxes = [], [], [], []
        for face_input, snap, bbox in faces:
            emb = extract_embedding(face_input)
            dets.append((bbox, 0.99, 'face'))
            embeds.append(emb)
            snapshots.append(snap)
            bboxes.append(np.array(bbox))
        if len(dets) != len(embeds):
            continue
        tracks = tracker.update_tracks(dets, embeds=embeds, frame=frame)
        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            tid = track.track_id
            bbox_track = np.array(track.to_ltwh())
            best_idx, best_dist = -1, float("inf")
            for idx, bbox_face in enumerate(bboxes):
                dist = np.linalg.norm(bbox_track - bbox_face)
                if dist < best_dist:
                    best_idx, best_dist = idx, dist
            if best_idx == -1 or best_dist > 20:
                continue
            emb = embeds[best_idx]
            snap = snapshots[best_idx]
            query = emb.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query)
            D, I = index.search(query, k=1)
            sim = float(D[0][0])
            name = names[I[0][0]] if sim > SIM_THRESHOLD else register_new_face(emb, snap)
            if tid not in track_info:
                track_info[tid] = {"name": name, "appear_count": 1, "last_seen_frame": f_id}
                log_event(tid, name, "appear", snap)
            else:
                track_info[tid]["last_seen_frame"] = f_id
                track_info[tid]["appear_count"] += 1
        for tid in list(track_info):
            if f_id - track_info[tid]["last_seen_frame"] > disappear_timeout:
                log_event(tid, track_info[tid]["name"], "disappear")
                del track_info[tid]
        tracking_queue.put((frame.copy(), tracks))

def thread_display():
    cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Tracking", 1280, 720)
    while not stop_event.is_set():
        try:
            frame, tracks = tracking_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        rx, ry, rw, rh = ROI
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, w, h = map(int, track.to_ltwh())
            name = track_info.get(track.track_id, {}).get("name", "?")
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}_{track.track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()
            break
    cv2.destroyAllWindows()

# ========== Launch ==========
threading.Thread(target=thread_camera_detect, daemon=True).start()
threading.Thread(target=thread_embedding, daemon=True).start()
threading.Thread(target=thread_tracking, daemon=True).start()
threading.Thread(target=thread_display, daemon=True).start()

try:
    while not stop_event.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_event.set()

cv2.destroyAllWindows()