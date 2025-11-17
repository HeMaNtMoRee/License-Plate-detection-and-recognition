import os
import traceback
from PIL import Image, UnidentifiedImageError
from paddleocr import PaddleOCR
from utils.json_ import process_json_data, log_plate_result
import numpy as np
import time
from utils.auth_checker import check_plate_authorization

# ---------------- CONFIG ----------------
MIN_SIZE = 640

ocr = PaddleOCR(
    lang='en',
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
)

# ---------------- IMAGE RESIZE FUNCTION ----------------
def resize_image_keep_aspect(image_path, min_size=MIN_SIZE):
    """Resize image so shortest side is at least min_size, keep aspect ratio."""
    try:
        img = Image.open(image_path)
        w, h = img.size

        if min(w, h) >= min_size:
            return img

        if w < h:
            new_w = min_size
            new_h = int(h * (min_size / w))
        else:
            new_h = min_size
            new_w = int(w * (min_size / h))

        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        return img_resized

    except UnidentifiedImageError:
        pass
    except Exception as e:
        pass
    return None


# # ---------------- MAIN FUNCTION ----------------
def plate_ocr(raw_img_path=None):
    try:
        for filename in os.listdir(raw_img_path):
            file_path = os.path.join(raw_img_path, filename)

            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    image = resize_image_keep_aspect(file_path)  # Returns PIL image
                    if image is None:
                        continue
                    
                    if image.mode not in ("RGB", "L"):   # convert everything except grayscale (L)
                        image = image.convert("RGB")

                    image_np = np.array(image)

                    # Handle grayscale (single channel) → expand to 3-channel
                    if image_np.ndim == 2:
                        image_np = np.stack([image_np]*3, axis=-1)

                    try:
                        result = ocr.predict(image_np)
                        time_ocr=time.time()                        
                        
                    except Exception as e:
                        print(f"OCR prediction failed for {filename}: {e} ")
                        continue

                    for i, res in enumerate(result):
                        try:
                            format_data = process_json_data(res)
                            auth_status = check_plate_authorization(format_data)
                            log_plate_result(format_data, auth_status)

                            print("OCR Result:", format_data)
                            print("Authorization:", auth_status.upper())
                            print("-" * 40)
                            
                        except Exception as e:
                            print(f"Error saving result for {filename}: {e}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    except Exception as e:
        print(f"Main OCR function error: {e}")