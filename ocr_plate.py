import os
import traceback
from PIL import Image, UnidentifiedImageError
from paddleocr import PaddleOCR
from json_ import process_json_data
import numpy as np
import time
# ---------------- CONFIG ----------------
MIN_SIZE = 640

ocr = PaddleOCR(
    lang='en',
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    # text_det_limit_side_len=320,  # Increase side length for better detection
    # text_det_thresh=0.6, #Threshold for text detection model’s pixel classification.
    # text_det_box_thresh=0.7,
    # text_rec_score_thresh=0.6 ,# Threshold for text recognition confidence after detection,
    # # text_det_unclip_ratio=1.5,
    # use_doc_unwarping=True,
    # use_doc_orientation_classify=True,
    # ocr_version='PP-OCRv5',
    # enable_mkldnn=True,
    # use_textline_orientation=True
)



# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',
#     text_det_limit_side_len=960,
#     text_det_limit_type='max',
#     text_det_thresh=0.6, #fine-tuned for low false negatives increase text_det_thresh to 0.5–0.6 to skip low-confidence boxes → slightly faster.
#     # text_det_box_thresh=0.5,
#     # text_det_unclip_ratio=1.2,
#     text_rec_score_thresh=0.5,
#     use_doc_unwarping=False,
#     use_doc_orientation_classify=False
    # use_textline_orientation=True
# )
# ocr = PaddleOCR(
#     use_angle_cls=True,          # Detect rotated plates (0°, 180°)
#     lang='en',                   # English letters + digits on plates
#     # text_det_limit_side_len=960, # Resize image max side to 960px for detection
#     text_det_limit_type='max',   # Scale by the maximum side (prevents distortion)
#     text_det_thresh=0.3,         # Lower threshold to detect faint/small text
#     # text_det_box_thresh=0.5,     # Filter out low-confidence detection boxes
#     text_det_unclip_ratio=1.5,   # Expand detected boxes slightly to include full plate text

#     # Text Recognition Parameters
#     # text_recognition_model_name='CRNN',  # Recognition model; CRNN works well for plates
#     text_rec_score_thresh=0.5,           # Ignore re
# cognition results with very low confidence
#     # text_rec_input_shape=(3, 32, 320),   # Input image size for recognition (channels, height, width)

# )


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
            # print("ffffffff",filename)

            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    image = resize_image_keep_aspect(file_path)  # Returns PIL image
                    if image is None:
                        continue
                    
                    # if image.mode == 'RGBA':
                    #     image = image.convert('RGB')

                    # Convert PIL image to numpy array for OCR
                    # image_np = np.array(image)
                    
                    if image.mode not in ("RGB", "L"):   # convert everything except grayscale (L)
                        image = image.convert("RGB")

                    image_np = np.array(image)

                    # Handle grayscale (single channel) → expand to 3-channel
                    if image_np.ndim == 2:
                        image_np = np.stack([image_np]*3, axis=-1)

                    try:
                        # print("ocr running")
                        # print(image_np.sum())
                        result = ocr.predict(image_np)
                        time_ocr=time.time()
                        # print("result",result)
                        
                        
                    except Exception as e:
                        print(f"OCR prediction failed for {filename}: {e} ")
                        continue

                    for i, res in enumerate(result):
                        try:
                            format_data=process_json_data(res)
                            # return format_data
                            print(format_data)
                            
                            # Also print number plate text
                        except Exception as e:
                            print(f"Error saving result for {filename}: {e}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    except Exception as e:
        print(f"Main OCR function error: {e}")
        
# plate_ocr(r'D:\number plate detection\plate')