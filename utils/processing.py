import os
import json
import numpy as np
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import traceback
from utils.ocr_plate import resize_image_keep_aspect, ocr
from utils.json_ import process_json_data


# Ensure directory exists
UPLOAD_FOLDERS = {
    'authorized': os.path.join('license_plates', 'authorized'),
    'blacklisted': os.path.join('license_plates', 'blacklisted'),
    'registered_plates': os.path.join('registered_plates', 'registered_plates.json'),
    'logs': os.path.join('logs', 'upload_events.jsonl')
}

# Create folders and files if missing
for key, path in UPLOAD_FOLDERS.items():
    folder = os.path.dirname(path) if path.endswith('.json') or path.endswith('.jsonl') else path

    # create directory
    os.makedirs(folder, exist_ok=True)

    # create empty file if it's meant to be a file
    if path.endswith('.json') and not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)   # start with empty json object

    if path.endswith('.jsonl') and not os.path.exists(path):
        open(path, 'w').close()  # create empty .jsonl file

def log_upload_event(filename: str, upload_type: str):
    """
    Writes a log of the file upload event to upload_events.jsonl.
    """
    try:
        now = datetime.now()
        log_entry = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
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
        # 1. Resize/Open image
        image_pil = resize_image_keep_aspect(image_path)
        if image_pil is None:
            return {"error": "Could not open or resize image."}

        # 2. Convert to NumPy array for OCR
        if image_pil.mode not in ("RGB", "L"):
            image_pil = image_pil.convert("RGB")
        
        image_np = np.array(image_pil)

        if image_np.ndim == 2:  # Handle grayscale
            image_np = np.stack([image_np] * 3, axis=-1)

        # 3. Run OCR prediction
        result = ocr.predict(image_np)
        
        if not result or not result[0]:
            print("[INFO] OCR returned no result.")
            return {"error": "OCR could not detect text."}

        # 4. Process the result
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


def _read_plates_db() -> dict:
    """Reads the registered_plates.json file and returns its content."""
    data = {"plates": []}
    try:
        if os.path.exists(REGISTERED_PLATES_FILE):
            with open(REGISTERED_PLATES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "plates" not in data or not isinstance(data["plates"], list):
                    print(f"[WARN] 'plates' key missing or not a list in {REGISTERED_PLATES_FILE}. Resetting.")
                    data = {"plates": []}
    except json.JSONDecodeError:
        print(f"[WARN] {REGISTERED_PLATES_FILE} is corrupted. Creating a new one.")
        data = {"plates": []}
    return data

def _write_plates_db(data: dict):
    """Writes the data object back to registered_plates.json."""
    try:
        with open(REGISTERED_PLATES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Successfully updated {REGISTERED_PLATES_FILE}.")
    except Exception as e:
        print(f"[ERROR] Failed to write to {REGISTERED_PLATES_FILE}: {e}")


def update_registered_plates(plate_data: dict, status: str):
    """
    Adds a SINGLE new plate data to registered_plates.json with the given status.
    """
    if not plate_data.get("license_plate"):
        print("[WARN] No license_plate data found, skipping update.")
        return

    # 1. Add the requested status to the data
    plate_data["status"] = status
    
    # 2. Read the existing data
    data = _read_plates_db()

    # 3. Append the new plate data
    data["plates"].append(plate_data)

    # 4. Write the file back
    _write_plates_db(data)


def add_plates_from_json_file(json_file_path: str, status: str) -> dict:
    """
    Reads a JSON file containing a list of plates and adds them to
    registered_plates.json with the given status.
    """
    try:
        # 1. Read the UPLOADED JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
    except json.JSONDecodeError:
        return {"error": "Uploaded file is not a valid JSON."}
    except Exception as e:
        return {"error": f"Failed to read uploaded file: {e}"}

    # 2. Determine if the JSON is a list or a dict with a 'plates' key
    new_plates_list = []
    if isinstance(new_data, list):
        new_plates_list = new_data
    elif isinstance(new_data, dict) and "plates" in new_data:
        new_plates_list = new_data["plates"]
    else:
        return {"error": "JSON format not recognized. Expected a list of plates, or a dictionary like {'plates': [...] }."}

    if not new_plates_list:
        return {"info": "No plates found in the JSON file."}

    # 3. Read the master database
    master_data = _read_plates_db()

    # 4. Process and append new plates
    count = 0
    for plate_entry in new_plates_list:
        if isinstance(plate_entry, dict) and plate_entry.get("license_plate"):
            plate_entry["status"] = status  # Set the status
            master_data["plates"].append(plate_entry)
            count += 1
        else:
            print(f"[WARN] Skipping invalid entry in JSON: {plate_entry}")

    # 5. Write back to the master database ONCE
    _write_plates_db(master_data)
    
    success_message = f"Successfully added {count} plates."
    print(f"[INFO] {success_message} from {json_file_path}")
    return {"success": success_message}