import cv2
import numpy as np

def align_face(image, landmarks, output_size=(112, 112)):
    ref = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    src = np.array(landmarks, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)[0]
    return cv2.warpAffine(image, M, output_size)

def preprocess_face(face):
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))  # CHW
    return np.expand_dims(face, axis=0).astype(np.float32)

def preprocess_input(image, size=640):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w, pad_h = size - new_w, size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114,114,114))
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1) / 255.0
    return np.expand_dims(img, axis=0), scale, (left, top)

def compute_iou(box, boxes):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / (area1 + area2 - inter + 1e-6)

def non_max_suppression(boxes, scores, landmarks, conf_thres=0.6, iou_thres=0.5):
    boxes = boxes[0]
    scores = scores[0][:, 0]
    landmarks = landmarks[0]
    mask = scores > conf_thres
    boxes, scores, landmarks = boxes[mask], scores[mask], landmarks[mask]
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    indices = np.argsort(scores)[::-1]
    boxes_xyxy, scores, landmarks = boxes_xyxy[indices], scores[indices], landmarks[indices]
    result = []
    while len(boxes_xyxy):
        result.append((boxes_xyxy[0], landmarks[0]))
        if len(boxes_xyxy) == 1: break
        iou = compute_iou(boxes_xyxy[0], boxes_xyxy[1:])
        keep = iou < iou_thres
        boxes_xyxy, scores, landmarks = boxes_xyxy[1:][keep], scores[1:][keep], landmarks[1:][keep]
    return result
