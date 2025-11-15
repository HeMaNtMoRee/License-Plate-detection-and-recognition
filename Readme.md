# License Plate Detection and OCR System

A comprehensive automated number plate recognition (ANPR) system that detects license plates using TensorFlow object detection and performs Optical Character Recognition (OCR) with PaddleOCR. The system includes plate authorization checking against a registered database.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Key Features](#key-features)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This system performs real-time license plate detection and recognition with the following capabilities:

- **Real-time Video Processing**: Captures video from webcam and detects number plates
- **Deep Learning Detection**: Uses TensorFlow trained model for robust plate detection
- **Optical Character Recognition**: Extracts text from plates using PaddleOCR
- **Plate Validation**: Validates detected plates against registered database
- **Multi-emirate Support**: Recognizes UAE license plates from all emirates (Abu Dhabi, Dubai, Sharjah, Ajman, Fujairah, Ras Al Khaimah, Umm Al Quwain)
- **Smart Filtering**: Uses sharpness analysis and stability checks to process only high-quality detections
- **Async Processing**: Runs OCR in background threads to maintain smooth video playback

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Video Input (Webcam)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            TensorFlow Object Detection Model                │
│     (Detects license plates in each frame)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Stability & Quality Checks                     │
│    - IoU tracking for temporal consistency                  │
│    - Laplacian sharpness analysis                           │
│    - Buffer management (BUFFER_SIZE frames)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Image Enhancement & Preprocessing                │
│    - Upscaling (2x), denoising, contrast adjustment         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         PaddleOCR (Background Thread)                       │
│    - Extracts text from enhanced plate images               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          JSON Data Processing & Parsing                     │
│    - Tokenizes OCR output                                   │
│    - Detects region (emirate)                               │
│    - Extracts category & number                             │
│    - Logs results to JSONL file                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Authorization Checker                               │
│    - Checks against registered_plates.json                  │
│    - Returns: authorized, blacklisted, or unauthorized      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Final Output                              │
│    - Console display, logs, and database storage            │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### Step 1: Create Virtual Environment (Python 3.10.0)

Ensure Python 3.10.0 is installed on your system:

```bash
# Create a virtual environment named "venv"
python3.10 -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install PaddlePaddle (CPU version)

Inside the activated virtual environment:

```bash
python -m pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Check installed package versions:

```bash
pip list
```

Expected versions:
- `tensorflow`: 2.13.0
- `paddleocr`: 3.1.0
- `opencv-python`: 4.7.0.72 or 4.12.0.88
- `numpy`: 1.24.3
- `paddlepaddle`: 3.1.0

If versions don't match, reinstall with specific versions:

```bash
pip install tensorflow==2.13.0 opencv-python==4.7.0.72 paddleocr==3.1.0 numpy==1.24.3 --force-reinstall
```

### Step 5: Verify Model Files

Ensure the following directories exist in your project root:

```
exported_model/
├── saved_model/
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
└── checkpoint/
```

### Step 6: Create Required Folders

```bash
mkdir cropped_number_plate_images
```

### Step 7: Configure Registered Plates Database

Edit `registered_plates/registered_plates.json` to add authorized license plates:

```json
{
    "plates": [
        {
            "license_plate": "2 Sharjah 14567",
            "category": "2",
            "region": "Sharjah",
            "number": "14567",
            "status": "authorized"
        },
        {
            "license_plate": "DD Dubai 50748",
            "category": "DD",
            "region": "Dubai",
            "number": "50748",
            "status": "blacklisted"
        }
    ]
}
```

---

## Configuration

### Main Script Configuration (`version3.py`)

Key configuration parameters at the top of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.6 | Minimum detection confidence to process |
| `STABLE_FRAMES` | 5 | Consecutive frames with IoU > threshold to trigger OCR |
| `IOU_THRESH` | 0.5 | Intersection-over-Union threshold for tracking |
| `BUFFER_SIZE` | 10 | Number of frames to buffer for quality assessment |
| `MIN_SHARPNESS` | 120.0 | Minimum Laplacian sharpness score to process |
| `OCR_COOLDOWN_FRAMES` | 30 | Frames to wait between OCR processing |
| `PADDING` | 20 | Pixels to pad around detected plate bbox |
| `FRAME_WIDTH` | 640 | Camera frame width |
| `FRAME_HEIGHT` | 480 | Camera frame height |
| `FPS` | 30 | Target frames per second |

Adjust these values based on your camera quality and processing speed:
- Lower `CONFIDENCE_THRESHOLD` for more detections (may increase false positives)
- Lower `MIN_SHARPNESS` for lower-quality cameras
- Increase `OCR_COOLDOWN_FRAMES` to reduce processing load

---

## How It Works

### Phase 1: Real-time Detection

1. **Video Capture**: Frames continuously captured from webcam at 30 FPS
2. **TensorFlow Inference**: Each frame passed to trained SSD model
3. **Bounding Box Extraction**: Model outputs detected plate locations with confidence scores
4. **Detection Filtering**: Only detections with confidence > `CONFIDENCE_THRESHOLD` proceed

### Phase 2: Stability & Quality Assessment

1. **Temporal Tracking**: Current bounding box compared to previous using IoU metric
   - If IoU > `IOU_THRESH`, increment `stable_count`
   - If IoU < threshold, reset count to 1
2. **Buffer Management**: Each detected plate crop added to circular buffer
3. **Sharpness Analysis**: Laplacian operator calculates focus quality of each crop
   - Highest quality crop retained from buffer
4. **Quality Threshold**: Only processes if sharpness ≥ `MIN_SHARPNESS`

### Phase 3: Image Enhancement

Cropped plate images enhanced for OCR accuracy:

1. **Upscaling**: 2x bilinear interpolation
2. **Denoising**: Non-local means denoising (7,7,7,21 parameters)
3. **Sharpening**: High-pass filter combination with alpha=1.25, beta=15
4. **Contrast Enhancement**: Additive boost for better text visibility

Code snippet from `enhance_for_ocr()`:
```python
- Resize: 2x upscaling
- Denoise: FastNlMeansDenoisingColored (removes noise while preserving edges)
- Sharpen: Unsharp mask technique (1.6x original - 0.6x blurred)
- Scale: Brightness and contrast adjustment
```

### Phase 4: Optical Character Recognition

1. **PaddleOCR Processing**: Enhanced image processed by PaddleOCR model
   - Language: English
   - Doc unwarping: Disabled (plates already relatively flat)
2. **Text Extraction**: Model outputs:
   - Recognition texts (OCR tokens)
   - Recognition scores (confidence per token)

### Phase 5: JSON Processing & Parsing

The `json_.py` module performs intelligent parsing:

1. **Tokenization**: Raw OCR strings split into meaningful tokens
   - Handles various separators: commas, dashes, slashes, colons
   - Preserves dotted abbreviations (e.g., "R.A.K.", "U.A.Q.")

2. **Region Detection** (3-level strategy):
   - **Level 1 - Exact Alias Match**: Direct match to alias list (confidence: 0.99)
   - **Level 2 - Cleaned Match**: Match after removing dots/spaces (confidence: varies)
   - **Level 3 - Fuzzy Matching**: Difflib similarity for typos/OCR errors (confidence: 0.6-0.95)

3. **Category Extraction**: 
   - Uses region-specific validation rules
   - Abu Dhabi: digits 1, 4-21, or 50
   - Dubai: 1-2 letter codes OR 1-2 digits
   - Sharjah: digits 1-5
   - Ajman: A, B, C, D, E, H
   - Fujairah: A-T (various letters)
   - Ras Al Khaimah: Specific letter codes
   - Umm Al Quwain: A-X letters

4. **Number Extraction**:
   - Extracts longest contiguous digit sequence (1-5 digits for UAE plates)
   - Filters out sequences > 5 digits
   - Confidence based on purity and length

5. **Logging**: Results saved to `logs/plate_logs.jsonl` with timestamp

### Phase 6: Authorization Checking

The `auth_checker.py` module validates against database:

1. **Database Load**: Reads `registered_plates/registered_plates.json`
2. **Matching**: Compares extracted (region, category, number) triplet
3. **Status Return**:
   - `"authorized"`: Plate in database with authorized status
   - `"blacklisted"`: Plate in database with blacklisted status
   - `"unauthorized"`: Plate not found in database

---

## Project Structure

```
py_env/
├── version3.py                          # Main detection script
├── requirements.txt                     # Python dependencies
├── Readme.md                            # This file
│
├── utils/
│   ├── ocr_plate.py                    # PaddleOCR wrapper
│   ├── json_.py                        # JSON parsing
│   ├── auth_checker.py                 # Authorization validation
│   ├── detection_utils_new2.py         # Utility functions (not used in app.py)
│
│
├── exported_model/                      # TensorFlow model files
│   ├── saved_model/
│   │   ├── saved_model.pb
│   │   ├── fingerprint.pb
│   │   ├── assets/
│   │   └── variables/
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   └── checkpoint/
│
├── cropped_number_plate_images/         # Temporary storage for detected plates
│   └── plate_latest.png                # Latest captured plate image
│
├── registered_plates/
│   └── registered_plates.json           # Database of authorized/blacklisted plates
│
├── logs/
    └── plate_logs.jsonl                 # Detection history (JSONL format)
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main entry point - video capture, detection, stability checks, async OCR |
| `ocr_plate.py` | Wraps PaddleOCR model, iterates through cropped images |
| `json_.py` | Core parsing logic - tokenization, region/category/number extraction |
| `auth_checker.py` | Database lookup for authorization status |
| `detection_utils_new2.py` | Legacy utility functions (face detection, SSD detection) |

---

## Usage

### Running the Application

```bash
# With virtual environment activated:
python app.py
```

**What happens:**
1. Video window opens showing live webcam feed
2. Detected plates shown with blue bounding boxes
3. When stable plate detected:
   - Image saved to `cropped_number_plate_images/plate_latest.png`
   - OCR runs in background thread
   - Results printed to console
4. Output displayed:
   ```
   OCR Result: {
       'license_plate': '2 Sharjah 14567',
       'region': 'Sharjah',
       'category': '2',
       'number': '14567',
       'confidences': {'region': 0.99, 'category': 0.9, 'number': 0.95}
   }
   Authorization: AUTHORIZED
   ```

### Exiting

Press **'q'** key to exit the application gracefully.

### Viewing Logs

Check detection history:

```bash
# View last 10 entries:
tail -n 10 logs/plate_logs.jsonl

# Pretty-print with jq (if installed):
cat logs/plate_logs.jsonl | jq .
```

Log format (JSONL - one JSON object per line):
```json
{"date": "2025-11-15", "time": "14:30:45", "license_plate": "2 Sharjah 14567", "category": "2", "region": "Sharjah", "number": "14567", "confidences": {"region": 0.99, "category": 0.9, "number": 0.95}}
```

---

<!-- ## Key Features

### 1. Robust Detection Pipeline
- Multi-stage filtering prevents false positives
- Stability tracking ensures consistent detections
- Quality-based processing only uses sharp, clear images

### 2. Intelligent OCR Parsing
- Handles poor OCR output (typos, missing characters)
- Supports all UAE emirate formats
- Fuzzy matching tolerates OCR errors
- Canonical region names with multiple aliases

### 3. Asynchronous Processing
- OCR runs in background thread
- Video playback remains smooth
- Processing cooldown prevents redundant OCR

### 4. Comprehensive Logging
- JSONL format for easy analysis
- Timestamps for all detections
- Confidence scores for validation

### 5. Database Integration
- Simple JSON-based authorized plates database
- Status tracking: authorized, blacklisted, unauthorized
- Easy to extend with API integration

--- -->

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails
```
[INFO] Loading TensorFlow model...
ERROR: Model not found at ...
```
**Solution**: Verify `exported_model/saved_model/` directory exists with all files.

#### 2. Camera Not Opening
```
Error: Could not open webcam
```
**Solution**:
- Check camera connection
- Try different camera index: `cv2.VideoCapture(1)` instead of 0
- Verify no other app is using the camera

#### 3. PaddleOCR Takes Too Long
**Solution**:
- First run downloads model (~200MB) - this is normal
- Increase `OCR_COOLDOWN_FRAMES` to reduce frequency
- Check `MIN_SHARPNESS` - lower values trigger more OCRs

#### 4. Poor Detection Accuracy
**Solutions**:
- Adjust `CONFIDENCE_THRESHOLD` down to 0.4-0.5
- Increase `STABLE_FRAMES` to 8-10 for more consistent detections
- Improve lighting conditions
- Clean camera lens

#### 5. OCR Not Working
```
Processing OCR in background thread...
ERROR: OCR: ...
```
**Solution**: Verify PaddleOCR installation:
```bash
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(lang='en'); print('OK')"
```

#### 6. Authorization Always Returns "unauthorized"
**Solution**: Verify:
- Plates in database match detected format exactly
- JSON file has correct schema with all required fields
- Database path is correct: `registered_plates/registered_plates.json`

### Performance Optimization

**For Low-End Hardware:**
```python
# In version3.py, increase these:
CONFIDENCE_THRESHOLD = 0.7      # Be more selective
STABLE_FRAMES = 10              # Wait for more consistency
MIN_SHARPNESS = 150.0           # Only very sharp images
OCR_COOLDOWN_FRAMES = 60        # Reduce OCR frequency
```
<!-- 
**For High-Volume Processing:**
```python
# Consider:
- Using GPU acceleration (CUDA/cuDNN) for TensorFlow
- Batch processing multiple frames
- Deploying on edge device (NVIDIA Jetson, etc.)
``` -->

---

## Dependencies Summary

Key libraries used:

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.13.0 | Object detection model inference |
| OpenCV | 4.7.0+ | Video capture, image processing |
| PaddleOCR | 3.1.0 | Optical character recognition |
| PaddlePaddle | 3.1.0 | Deep learning framework for PaddleOCR |
| NumPy | 1.24.3 | Numerical computations |

---

## Support & Maintenance

For issues or questions:
1. Check Troubleshooting section above
2. Verify all dependencies installed correctly
3. Check log files in `logs/` directory
4. Ensure model files are present and valid

---

Last Updated: November 2025
