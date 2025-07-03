import os
import cv2
import numpy as np
import onnxruntime
from utils.face_utils import align_face, preprocess_face, preprocess_input, non_max_suppression

# ==== CẤU HÌNH ====
IMAGE_FOLDER = "imges_embedding"
YOLO_MODEL_PATH = "models/yolov5n-0.5.onnx"
EMBED_MODEL_PATH = "models/webface_r50.onnx"
EMBED_SAVE_PATH = "embeddings/embeddings.npy"
NAMES_SAVE_PATH = "embeddings/names.npy"

# ==== Load mô hình ====
yolo_session = onnxruntime.InferenceSession(YOLO_MODEL_PATH)
embed_session = onnxruntime.InferenceSession(EMBED_MODEL_PATH)

yolo_input_name = yolo_session.get_inputs()[0].name
embed_input_name = embed_session.get_inputs()[0].name

# ==== Hàm trích xuất embedding từ 1 ảnh ====
def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Không đọc được ảnh: {image_path}")
        return None

    input_blob, scale, pad = preprocess_input(img, size=640)

    boxes, scores, landmarks = yolo_session.run(None, {yolo_input_name: input_blob})
    detections = non_max_suppression(boxes, scores, landmarks, conf_thres=0.6, iou_thres=0.5)

    if len(detections) == 0:
        print(f"[!] Không phát hiện khuôn mặt: {image_path}")
        return None

    # Dùng khuôn mặt đầu tiên
    (box, lmk) = detections[0]

    # Giải scale & padding về kích thước ảnh gốc
    lmk = np.array(lmk).reshape(5, 2)
    lmk[:, 0] -= pad[0]
    lmk[:, 1] -= pad[1]
    lmk /= scale

    face_aligned = align_face(img, lmk)
    if face_aligned is None:
        print(f"[!] align_face thất bại: {image_path}")
        return None

    face_input = preprocess_face(face_aligned)  # shape: (1, 3, 112, 112)

    # Trích xuất embedding
    embedding = embed_session.run(None, {embed_input_name: face_input})[0]  # shape: (1, 512)
    return embedding[0]  # shape: (512,)

# ==== Duyệt ảnh và lưu ====
embeddings = []
names = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        name = os.path.splitext(filename)[0]
        emb = extract_embedding(image_path)
        if emb is not None:
            embeddings.append(emb)
            names.append(name)

# ==== Lưu vào file ====
embeddings = np.array(embeddings).astype(np.float32)
names = np.array(names)

np.save(EMBED_SAVE_PATH, embeddings)
np.save(NAMES_SAVE_PATH, names)

print(f"✅ Đã lưu {len(embeddings)} embedding vào {EMBED_SAVE_PATH}")
print(f"✅ Tên tương ứng lưu vào {NAMES_SAVE_PATH}")
