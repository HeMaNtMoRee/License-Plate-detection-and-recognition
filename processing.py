import os
import json
import numpy as np
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import traceback

# Import the components from your existing utils
try:
    from utils.ocr_plate import resize_image_keep_aspect, ocr
    from utils.json_ import process_json_data
except ImportError:
    print("[ERROR] Could not import from utils. Ensure utils/ocr_plate.py and utils/json_.py exist.")
    exit(1)

REGISTERED_PLATES_FILE = os.path.join('registered_plates', 'registered_plates.json')
UPLOAD_LOG_FILE = os.path.join('logs', 'upload_events.jsonl')

# Ensure log directory exists
os.makedirs('logs', exist_ok=True)


def log_upload_event(filename: str, upload_type: str):
    """
    Writes a log of the file upload event to upload_events.jsonl.
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "upload_type": upload_type
        }
        with open(UPLOAD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to log upload event: {e}")


def extract_data_from_image(image_path: str) -> dict:
    """
    Runs the OCR process on a single image file and returns the
    normalized plate data.
    """
    print(f"[INFO] Processing OCR for {image_path}...")
    try:
        # 1. Resize/Open image (reusing your function)
        image_pil = resize_image_keep_aspect(image_path)
        if image_pil is None:
            return {"error": "Could not open or resize image."}

        # 2. Convert to NumPy array for OCR
        if image_pil.mode not in ("RGB", "L"):
            image_pil = image_pil.convert("RGB")
        
        image_np = np.array(image_pil)

        if image_np.ndim == 2:  # Handle grayscale
            image_np = np.stack([image_np] * 3, axis=-1)

        # 3. Run OCR prediction (reusing your global ocr object)
        # We expect one result for the cropped plate
        result = ocr.predict(image_np)
        
        if not result or not result[0]:
            print("[INFO] OCR returned no result.")
            return {"error": "OCR could not detect text."}

        # 4. Process the result (reusing your json_ util)
        # result is like [[(box, (text, score))], ...]
        # We pass the first detected text block to process_json_data
        
        # Extract rec_texts and rec_scores for process_json_data
        ocr_data = {
            "rec_texts": result[0]["rec_texts"],
            "rec_scores": [result[0]['rec_scores']]
        }


        formatted_data = process_json_data(ocr_data)
        print(f"[INFO] OCR Result: {formatted_data}")
        return formatted_data

    except UnidentifiedImageError:
        print(f"[ERROR] Cannot identify image file: {image_path}")
        return {"error": "Cannot identify image file."}
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
        traceback.print_exc()
        return {"error": f"OCR extraction failed: {e}"}


def update_registered_plates(plate_data: dict, status: str):
    """
    Adds the new plate data to registered_plates.json with the given status.
    """
    if not plate_data.get("license_plate"):
        print("[WARN] No license_plate data found, skipping update.")
        return

    # 1. Add the requested status to the data
    plate_data["status"] = status
    
    # 2. Read the existing data
    data = {"plates": []}
    try:
        if os.path.exists(REGISTERED_PLATES_FILE):
            with open(REGISTERED_PLATES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "plates" not in data:
                    data = {"plates": []}
    except json.JSONDecodeError:
        print(f"[WARN] {REGISTERED_PLATES_FILE} is corrupted. Creating a new one.")
        data = {"plates": []}

    # 3. Append the new plate data
    data["plates"].append(plate_data)

    # 4. Write the file back
    try:
        with open(REGISTERED_PLATES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Successfully updated {REGISTERED_PLATES_FILE}.")
    except Exception as e:
        print(f"[ERROR] Failed to write to {REGISTERED_PLATES_FILE}: {e}")