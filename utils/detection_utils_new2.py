import os
import cv2
import numpy as np
import tensorflow as tf
import face_recognition
import mediapipe as mp
import pickle
import hashlib
from logger_setup import logger

# --- New: Path for the pickle file to store encodings ---
ENCODINGS_FILE = "face_encodings.pkl"

# Load the SSD Model
# NOTE: The model_dir path is specific to your local machine.
model_dir = "D:/Kody/AIORTC/weapon_detection_model"
detect_fn = tf.saved_model.load(model_dir)

# Face Recognition Variables
known_face_encodings = []
known_face_names = []
face_detection = None

# Class labels
# NOTE: Corrected the duplicate key for Pistol and Rifle
category_index = {
    1: {'id': 1, 'name': 'Person'},
    2: {'id': 2, 'name': 'Knife'},
    3: {'id': 3, 'name': 'Fight'},
    4: {'id': 4, 'name': 'Pistol'},
    5: {'id': 5, 'name': 'Rifle'},
}


def recognize_faces(frame, draw=False):
    global face_detection
    global category_encodings
    
    if not hasattr(recognize_faces, "initialized"):
        recognize_faces.initialized = False
        category_encodings = {}
        # New: Store the hash of the directory content
        recognize_faces.dir_hash = ""

    # --- New: Function to calculate a hash of the trainedImages directory ---
    def calculate_dir_hash(dir_path):
        """Calculates a hash based on file names, sizes, and modification times."""
        if not os.path.exists(dir_path):
            return ""
        hasher = hashlib.sha256()
        for root, dirs, files in os.walk(dir_path):
            # Sort to ensure consistent hash
            dirs.sort()
            files.sort()
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(root, filename)
                    hasher.update(filepath.encode('utf-8'))
                    hasher.update(str(os.path.getsize(filepath)).encode('utf-8'))
                    hasher.update(str(os.path.getmtime(filepath)).encode('utf-8'))
        return hasher.hexdigest()

    current_dir_hash = calculate_dir_hash(os.path.join(os.getcwd(), "trainedImages"))
    encodings_stale = True

    # --- New: Check for and load pre-computed encodings if the hash matches ---
    if os.path.exists(ENCODINGS_FILE):
        logger.debug(f"Found {ENCODINGS_FILE}. Checking for updates...")
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                saved_data = pickle.load(f)
                saved_encodings = saved_data['encodings']
                saved_hash = saved_data.get('hash', '')
            
            if saved_hash == current_dir_hash:
                category_encodings = saved_encodings
                recognize_faces.initialized = True
                encodings_stale = False
                logger.debug(f"Encodings are up-to-date. Loaded {len(category_encodings)} face categories.")
            else:
                logger.info("Directory hash has changed. Will re-encode from images.")

        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error loading or parsing {ENCODINGS_FILE}: {e}. Will re-encode from images.")

    def load_faces_from_dir(dir_path):
        """Loads encodings and names from a given directory."""
        encodings, names = [], []
        if not os.path.exists(dir_path):
            return encodings, names
        for filename in os.listdir(dir_path):
            try:
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                path = os.path.join(dir_path, filename)
                image = face_recognition.load_image_file(path)
                face_encs = face_recognition.face_encodings(image)
                if face_encs:
                    encodings.append(face_encs[0])
                    names.append(os.path.splitext(filename)[0])
            except Exception as e:
                logger.error(f"Error encoding {filename} in {dir_path}: {e}")
        return encodings, names

    # --- New: If encodings are stale, re-encode and save the new data and hash ---
    if encodings_stale:
        base_dir = os.path.join(os.getcwd(), "trainedImages")
        if os.path.exists(base_dir):
            for subdir in os.listdir(base_dir):
                subdir_path = os.path.join(base_dir, subdir)
                if os.path.isdir(subdir_path):
                    encs, nms = load_faces_from_dir(subdir_path)
                    if encs:
                        category_encodings[subdir.lower()] = (encs, nms)
                        logger.info(f"Loaded {len(encs)} faces from category: {subdir}")
        
        # Save the newly generated encodings and the current hash to a pickle file
        if category_encodings:
            try:
                with open(ENCODINGS_FILE, 'wb') as f:
                    saved_data = {'encodings': category_encodings, 'hash': current_dir_hash}
                    pickle.dump(saved_data, f)
                logger.info(f"Successfully saved new face encodings and hash to {ENCODINGS_FILE}")
            except (IOError, pickle.PickleError) as e:
                logger.error(f"Error saving face encodings to {ENCODINGS_FILE}: {e}")
        
        recognize_faces.initialized = True

    if face_detection is None:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    detection_data = []

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error("Failed to convert frame to RGB: %s", e)
        return detection_data, frame

    try:
        locations = face_recognition.face_locations(rgb_frame, model='hog')
        if not isinstance(locations, list):
            locations = list(locations) if locations is not None else []

        encodings = []
        if locations:
            encodings = face_recognition.face_encodings(rgb_frame, locations, num_jitters=0)

        for loc_idx, encoding in enumerate(encodings):
            top, right, bottom, left = locations[loc_idx]
            x, y = left, top
            w, h = right - left, bottom - top

            name = "Person"
            score = 0.0
            category_remark = "unauthorised"  # default

            # Compare with each category
            for category_name, (enc_list, name_list) in category_encodings.items():
                distances = face_recognition.face_distance(enc_list, encoding)
                if len(distances) > 0:
                    best_idx = np.argmin(distances)
                    # Use a threshold for matching
                    if distances[best_idx] < 0.6:  # Common threshold for face_recognition
                        name = name_list[best_idx]
                        score = float(max(0.0, min(1.0, 1.0 - distances[best_idx])))
                        category_remark = category_name
                        break  # stop after first match

            if draw:
                try:
                    label_text = f"{name} ({category_remark})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - text_height - 6), (x + text_width, y), (0, 255, 0), -1)
                    cv2.putText(frame, label_text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                except Exception:
                    pass

            detection_data.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'label': name,
                'score': float(score),
                'category_remark': category_remark
            })

    except Exception as e:
        logger.error("Face recognition error: %s", e)

    return detection_data, frame


def is_face_inside_person(face_box, person_box):
    fx, fy, fw, fh = face_box
    px, py, pw, ph = person_box
    
    # Face center point
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2
    
    # Person box bounds
    if px <= face_cx <= px + pw and py <= face_cy <= py + ph:
        return True
    return False


def run_inference(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]  # shape (1, H, W, 3)

    detections = detect_fn(input_tensor)

    # Extract fields
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    return boxes, classes, scores


def detect_objects_ssd(frame, score_threshold=0.5, draw=False):
    
    boxes, classes, scores = run_inference(frame)
    h, w, _ = frame.shape
    detection_data = []

    for i in range(len(scores)):
        if scores[i] >= score_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            
            x = int(xmin * w)
            y = int(ymin * h)
            w_box = int((xmax - xmin) * w)
            h_box = int((ymax - ymin) * h)
            
            label = category_index.get(classes[i], {'name': 'unknown'})['name']
            if label.lower() == "person":
                category = "Human"
            elif label.lower() == "fight":
                category = "Incident"
            else:
                category = "Weapon"
            
            if draw:
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - text_height - 6), (x + text_width, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            detection_data.append({
                'x': x,
                'y': y,
                'w': w_box,
                'h': h_box,
                'category': category,
                'label': label,
                'score': float(scores[i]),
                'remark': None
            })
    
    return detection_data, frame


def process_frame(frame, person_score_threshold=0.5, face_score_threshold=0.4):
    # 1. Run SSD detection
    object_detections, _ = detect_objects_ssd(frame, score_threshold=person_score_threshold)

    all_detections = []

    for obj in object_detections:
        det_category = obj['category']
        det_label = obj['label']
        det_remark = None 
        
        if det_label.lower() == 'person':
            # Crop person region from frame
            px, py, pw, ph = obj['x'], obj['y'], obj['w'], obj['h']
            
            # Ensure the crop is within the frame bounds
            py_end = min(py + ph, frame.shape[0])
            px_end = min(px + pw, frame.shape[1])
            person_crop = frame[py:py_end, px:px_end]
            
            if person_crop.size > 0:
                # Run face detection + recognition inside this person crop
                face_detections, _ = recognize_faces(person_crop)

                # If faces detected, attach best match to person
                if face_detections:
                    face_info = face_detections[0]
                    if face_info['score'] >= face_score_threshold:
                        det_label = face_info['label']
                        det_remark = face_info['category_remark']
                        det_category = "Human"
                        obj['score'] = face_info['score']
                    else:
                        det_label = "Person"
                        det_remark = "unauthorised"
                        det_category = "Human"
                else:
                    det_label = "Person"
                    det_remark = "unauthorised"
                    det_category = "Human"
        
        elif det_label.lower() in ["license plate", "number plate"]:
            det_category = "License Plate"
            det_remark = "gujarat"  # Placeholder until OCR added

        else:
            det_remark = None
            
        all_detections.append({
            'x': obj['x'],
            'y': obj['y'],
            'w': obj['w'],
            'h': obj['h'],
            'category': det_category,
            'label': det_label,
            'remark': det_remark,
            'score': obj['score']
        })

    # Draw detections
    for det in all_detections:
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        label = f"{det['label']} ({det.get('score', 0.0):.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_h - 6), (x + text_w, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return all_detections, frame

def main():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.info("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Failed to read frame.")
                break

            detections, pframe = process_frame(frame)

            logger.info(detections)

            cv2.imshow("Webcam - Face and Object Detection", pframe)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('e') or key == ord('E'):
                break

    except Exception as e:
        logger.error("Exception during video loop:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import logging
    # Create and configure logger
    logging.basicConfig(filename="detection_util.log",
                        format='[%(asctime)s]  %(name)s: %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S'
                        ,level=logging.INFO)

    # Creating an object
    logger = logging.getLogger()
    main()

