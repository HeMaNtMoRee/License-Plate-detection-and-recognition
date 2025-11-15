import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from ocr_plate import plate_ocr

# =================== Config =================== #
CONFIDENCE_THRESHOLD = 0.6
STABLE_FRAMES = 5
IOU_THRESH = 0.5
BUFFER_SIZE = 10
MIN_SHARPNESS = 120.0
OCR_COOLDOWN_FRAMES = 30
PADDING = 20

# =================== Paths =================== #
cwd_path = os.getcwd()
raw_img_path = os.path.join(cwd_path, "cropped_number_plate_images")
model_path = os.path.join(cwd_path, "exported_model", "saved_model")


os.makedirs(raw_img_path, exist_ok=True)

# =================== Utilities =================== #
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area + 1e-9)

def get_crop(image, bbox, padding=PADDING):
    x1, y1, x2, y2 = bbox
    h_img, w_img = image.shape[:2]
    return image[max(0, y1 - padding):min(h_img, y2 + padding),
                 max(0, x1 - padding):min(w_img, x2 + padding)]

def enhance_for_ocr(crop, alpha=1.25, beta=15):
    if crop.size == 0:
        return crop
    enhanced = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 7, 21)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)
    return cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

def focus_measure(img_bgr):
    if img_bgr.size == 0:
        return 0.0
    return cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def clear_folder(path):
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            os.remove(fp)

# =================== Model Load =================== #
print("[INFO] Loading TensorFlow model...")
detect_fn = tf.saved_model.load(model_path)
category_index = {1: {'id': 1, 'name': 'number_plate'}}

# =================== Camera Init =================== #
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# =================== State =================== #
stable_count = 0
last_bbox = None
crop_buffer = deque(maxlen=BUFFER_SIZE)
cooldown = 0

# =================== Processing Loop =================== #
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # Inference
        rgb_tensor = tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[tf.newaxis, ...]
        detections = detect_fn(rgb_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        # Best detection
        idx = np.argmax(scores)
        if scores[idx] >= CONFIDENCE_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[idx]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            bbox = (x1, y1, x2, y2)
            label = category_index.get(classes[idx], {'name': 'unknown'})['name']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {scores[idx]*100:.1f}%", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Stability check
            stable_count = stable_count + 1 if last_bbox and iou_xyxy(bbox, last_bbox) >= IOU_THRESH else 1
            last_bbox = bbox

            # Store crop in buffer
            crop = get_crop(frame, bbox)
            crop_buffer.append((crop, focus_measure(crop)))
        else:
            stable_count = 0
            last_bbox = None
            crop_buffer.clear()

        # OCR triggering
        if cooldown > 0:
            cooldown -= 1

        if stable_count >= STABLE_FRAMES and cooldown == 0 and crop_buffer:
            best_crop, sharpness = max(crop_buffer, key=lambda t: t[1])
            if sharpness >= MIN_SHARPNESS:
                clear_folder(raw_img_path)
                out_path = os.path.join(raw_img_path, "plate_latest.png")
                cv2.imwrite(out_path, enhance_for_ocr(best_crop))
                try:
                    plate_ocr(raw_img_path)
                    clear_folder(raw_img_path)

                except Exception as e:
                    print(f"[ERROR] OCR: {e}")
                cooldown = OCR_COOLDOWN_FRAMES
                crop_buffer.clear()
            else:
                cv2.putText(frame, f"Waiting focus... {sharpness:.0f}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Live Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
