# License Plate Detection and OCR System

A comprehensive automated number plate recognition (ANPR) system that detects license plates using TensorFlow object detection and performs Optical Character Recognition (OCR) with PaddleOCR. The system provides both real-time video processing via command-line and web-based upload functionality with Flask.

## Table of Contents

- [Project Overview](#project-overview)
- [Two Main Modes](#two-main-modes)
- [System Architecture](#system-architecture)
- [Setup Instructions](#setup-instructions)
- [How It Works Step-by-Step](#how-it-works-step-by-step)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Key Features](#key-features)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This system performs automated number plate recognition with two operational modes:

- **Real-time Video Processing**: Live webcam feed with plate detection (`app.py`)
- **Web-based Upload Interface**: Flask web app for batch processing and database management (`flask_app.py`)

### Core Capabilities

- **Deep Learning Detection**: Uses TensorFlow trained SSD model for robust plate detection
- **Optical Character Recognition**: Extracts text from plates using PaddleOCR
- **Plate Validation**: Validates detected plates against registered database
- **Multi-emirate Support**: Recognizes UAE license plates (Abu Dhabi, Dubai, Sharjah, Ajman, Fujairah, Ras Al Khaimah, Umm Al Quwain)
- **Smart Filtering**: Sharpness analysis and stability checks for high-quality detections only
- **Async Processing**: Background thread OCR keeps video playback smooth
- **Database Management**: JSON-based plate database with authorized/blacklisted categories
- **Comprehensive Logging**: JSONL format logs for all detections and uploads

## Two Main Modes

### Mode 1: Real-Time Video (`app.py`)

**Purpose**: Continuous monitoring and plate detection from live video feed

- Continuous video capture from webcam at 30 FPS (640x480)
- Live plate detection with bounding boxes
- Automatic OCR when stable detection achieved
- Background thread processing (non-blocking)
- Console output and JSON logging
- Quality-based filtering (sharpness + stability checks)

**Best for**: Surveillance, traffic monitoring, parking systems

### Mode 2: Web Upload (`flask_app.py`)

**Purpose**: Interactive image upload and database management

- Single image upload with OCR processing
- Batch JSON file import for bulk plate loading
- Categorize plates as authorized or blacklisted
- View OCR results in browser (JSON formatted)
- Automatic database updates
- Upload event logging

**Best for**: Administrative tasks, database management, user-facing applications

---

## System Architecture

### Real-Time Mode (app.py) Data Flow

1. **Video Capture** → OpenCV reads 30 FPS from webcam (640x480 BGR)
2. **Detection** → TensorFlow SSD model detects license plates
3. **Filtering** → Keep only detections with confidence > 0.6
4. **Tracking** → IoU-based stability check (5+ consecutive frames)
5. **Quality** → Laplacian sharpness analysis (threshold: 120.0)
6. **Enhancement** → 2x upscale, denoise, sharpen, contrast boost
7. **OCR** → PaddleOCR extracts text (background thread)
8. **Parsing** → JSON processing extracts region/category/number
9. **Auth Check** → Validate against registered_plates.json
10. **Logging** → Save results to plate_logs.jsonl

### Web Mode (flask_app.py) Data Flow

1. **Upload** → Browser file selection (image or JSON)
2. **Validation** → Check file type and format
3. **Processing** → OCR (image) or JSON parsing (bulk import)
4. **Database** → Add status field and append to registered_plates.json
5. **Logging** → Record upload event to upload_events.jsonl
6. **Response** → Display results in browser

---

## Setup Instructions

### Step 1: Install Python 3.10.0

```bash
python --version
# Expected: Python 3.10.0
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Activate - Windows
venv\Scripts\activate
# Activate - macOS/Linux
source venv/bin/activate
```

### Step 3: Install PaddlePaddle (CPU)

```bash
python -m pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `tensorflow==2.13.0` - Model inference
- `paddleocr==3.1.0` - OCR processing
- `opencv-python==4.7.0.72` - Image processing
- `numpy==1.24.3` - Numerical operations
- `flask==2.3.0+` - Web server
- `pillow==11.1.0` - Image I/O

### Step 5: Verify Model Files

```
exported_model/
├── saved_model/
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
└── checkpoint/
```

### Step 6: Create Required Directories

```bash
mkdir -p cropped_number_plate_images
mkdir -p registered_plates
mkdir -p logs
mkdir -p license_plates/authorized
mkdir -p license_plates/blacklisted
```

### Step 7: Initialize Database

Create `registered_plates/registered_plates.json`:

```json
{
    "plates": []
}
```

---

## How It Works Step-by-Step

### Real-Time Processing Pipeline (app.py)

#### Phase 1: Video Capture & Detection
1. Read frame from webcam (30 FPS)
2. Convert BGR to RGB
3. Run TensorFlow SSD inference
4. Extract: bounding boxes, classes, confidence scores
5. Filter by CONFIDENCE_THRESHOLD (default: 0.6)

#### Phase 2: Stability Tracking
1. Calculate IoU (Intersection-over-Union) between current and previous bbox
2. If IoU > IOU_THRESH (0.5): increment stable_count
3. Else: reset stable_count to 1
4. Wait until stable_count >= STABLE_FRAMES (5)

#### Phase 3: Quality Assessment
1. For each buffered crop, calculate Laplacian sharpness
2. Select highest sharpness crop from buffer
3. Check if sharpness > MIN_SHARPNESS (120)
4. Only proceed if quality threshold met

#### Phase 4: Image Enhancement
```
Original crop
    ↓ 2x Upscale (bicubic)
    ↓ FastNlMeansDenoising
    ↓ Unsharp Mask (1.6x - 0.6x blur)
    ↓ Contrast Adjustment (alpha=1.25, beta=15)
Enhanced image for OCR
```

#### Phase 5: OCR Processing (Async)
1. Run in background thread (non-blocking)
2. Initialize PaddleOCR with English language
3. Extract text tokens and confidence scores
4. Return: rec_texts[], rec_scores[]

#### Phase 6: JSON Parsing (json_.py)

**Tokenization**: Split OCR strings on separators (-,/,,:), normalize to uppercase

**Region Detection** (3-level fallback):
- Level 1: Exact alias match (confidence: 0.99)
  - Examples: "DUBAI"→Dubai, "SHJ"→Sharjah, "AD"→Abu Dhabi
- Level 2: Fuzzy token matching (confidence: 0.6-0.95)
  - Handles typos and OCR errors
- Level 3: No match (region=None)

**Category Validation** (emirate-specific):
- Abu Dhabi: 1, 4-21, 50
- Dubai: A-Z letters or 01-99
- Sharjah: 1-5
- Ajman: A, B, C, D, E, H
- Fujairah: A, B, C, D, E, F, G, K, M, P, R, S, T
- Ras Al Khaimah: A, C, D, I, K, M, N, S, V, Y
- Umm Al Quwain: A-X

**Number Extraction**: Longest contiguous digit sequence (1-5 digits)

**Output Format**:
```json
{
    "license_plate": "2 Sharjah 14567",
    "region": "Sharjah",
    "category": "2",
    "number": "14567",
    "confidences": {
        "region": 0.99,
        "category": 0.9,
        "number": 0.95
    }
}
```

#### Phase 7: Authorization Check (auth_checker.py)
1. Load registered_plates.json
2. Check if (region, category, number) triplet exists
3. Return: "authorized" | "blacklisted" | "unauthorized"

#### Phase 8: Logging (plate_logs.jsonl)
```json
{
    "date": "2025-11-22",
    "time": "14:30:45",
    "license_plate": "2 Sharjah 14567",
    "category": "2",
    "region": "Sharjah",
    "number": "14567",
    "confidences": {"region": 0.99, "category": 0.9, "number": 0.95}
}
```

### Web Mode Processing (flask_app.py)

**Image Upload Workflow**:
1. Select image file (PNG, JPG, BMP)
2. Choose status (authorized/blacklisted)
3. Server validates file type
4. Save to license_plates/{status}/
5. Log upload event to upload_events.jsonl
6. Run OCR via extract_data_from_image()
7. Add status field to OCR result
8. Append to registered_plates.json
9. Display result as JSON in browser

**JSON Bulk Upload Workflow**:
1. Select JSON file with plate list
2. Choose status (authorized/blacklisted)
3. Server parses JSON structure
4. Validate each plate entry
5. Add status field to each entry
6. Read current registered_plates.json
7. Append all new plates
8. Atomic write (single operation)
9. Log upload event
10. Display count of added plates

---

## Project Structure

```
py_env/
├── app.py                           # Real-time video mode
├── flask_app.py                     # Web upload mode
├── requirements.txt                 # Python dependencies
├── Readme.md                        # This file
│
├── utils/
│   ├── ocr_plate.py                # PaddleOCR wrapper
│   ├── json_.py                    # JSON parsing & tokenization
│   ├── auth_checker.py             # Authorization validation
│   └── processing.py               # Flask helper functions
│
├── exported_model/                  # TensorFlow saved model
│   ├── saved_model/
│   │   ├── saved_model.pb
│   │   ├── fingerprint.pb
│   │   ├── assets/
│   │   └── variables/
│   └── checkpoint/
│
├── cropped_number_plate_images/    # Temp folder for app.py
├── registered_plates/              # Master plate database
├── license_plates/                 # Web upload storage
│   ├── authorized/
│   └── blacklisted/
├── logs/                            # Log files
│   ├── plate_logs.jsonl            # app.py detections
│   └── upload_events.jsonl         # flask_app.py uploads
├── templates/                       # Flask web interface
│   └── index.html
└── __pycache__/
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Real-time detection from webcam |
| `flask_app.py` | Web server for image/JSON upload |
| `ocr_plate.py` | PaddleOCR wrapper |
| `json_.py` | Data parsing and normalization |
| `auth_checker.py` | Authorization lookup |
| `processing.py` | Flask helper functions |

---

## Usage Guide

### Real-Time Mode

```bash
python app.py
```

Expected output:
```
[INFO] Loading TensorFlow model...
[INFO] Camera initialized
Live Plate Detection
(Blue boxes around plates)
(Press 'q' to quit)
```

Console output when plate detected:
```
Processing OCR in background thread...
OCR Result: {
    'license_plate': '2 Sharjah 14567',
    'region': 'Sharjah',
    'category': '2',
    'number': '14567',
    'confidences': {'region': 0.99, 'category': 0.9, 'number': 0.95}
}
Authorization: AUTHORIZED
Elapsed time: 2.34 seconds
```

### Web Mode

```bash
python flask_app.py
```

Open browser: `http://localhost:5000`

Features:
- File upload with drag & drop
- Radio buttons: Authorized / Blacklisted
- Upload button
- Results displayed as JSON

### Viewing Logs

**Real-time detections (app.py)**:
```bash
tail -f logs/plate_logs.jsonl
cat logs/plate_logs.jsonl | jq .
```

**Upload history (flask_app.py)**:
```bash
tail -f logs/upload_events.jsonl
```

**Database**:
```bash
jq '.plates | length' registered_plates/registered_plates.json
jq '.plates[] | select(.status=="authorized")' registered_plates/registered_plates.json
```

---

## Key Features

1. **Dual-Mode Processing**: Real-time + web interface
2. **Intelligent OCR Parsing**: Handles poor output with fuzzy matching
3. **Emirate-Aware**: All 7 UAE emirates supported
4. **Async Threading**: Non-blocking background OCR
5. **Quality Filtering**: Sharpness + stability checks
6. **Comprehensive Logging**: JSONL format for easy analysis
7. **Confidence Scoring**: Every output has metrics
8. **Database Integration**: JSON-based (extensible to SQL)

---

## Troubleshooting

### Model Loading Fails
- Verify `exported_model/saved_model/` exists
- Check file permissions
- Reinstall TensorFlow: `pip install --force-reinstall tensorflow==2.13.0`

### Camera Not Opening
- Check physical connection
- Try camera index 1: `cv2.VideoCapture(1)`
- Close other camera apps
- Use: `cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)` on Windows

### PaddleOCR Slow (First Run)
- Normal: Downloads ~200MB model on first run (2-5 minutes)
- Subsequent runs are fast
- Monitor with: `tail -f logs/plate_logs.jsonl`

### OCR Not Detecting
- Ensure good image quality (sharp, well-lit)
- Lower MIN_SHARPNESS if needed
- Check image resolution (should be > 100x100)

### Authorization Always "unauthorized"
- Verify database has plates: `jq '.plates | length' registered_plates/registered_plates.json`
- Check region/category/number match exactly (case-sensitive region)
- Ensure JSON structure is correct

### Flask Port Already in Use
- Change port in flask_app.py: `app.run(port=5001)`
- Or kill process: `lsof -i :5000`

### Low Detection Accuracy
Tune parameters in app.py:
```python
CONFIDENCE_THRESHOLD = 0.7      # More selective
STABLE_FRAMES = 8               # More consistency
MIN_SHARPNESS = 150             # Better quality
```

---

**Last Updated**: November 2025 <br>
**Repository**: License-Plate-detection-and-recognition <br>
**Branch**: feature
