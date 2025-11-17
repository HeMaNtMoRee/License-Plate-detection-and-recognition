import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import the new processing functions
from processing import log_upload_event, extract_data_from_image, update_registered_plates

# --- Configuration ---
UPLOAD_FOLDERS = {
    'authorized': os.path.join('license_plates', 'authorized'),
    'blacklisted': os.path.join('license_plates', 'blacklisted')
}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Change this for production

# Ensure upload folders exist
os.makedirs(UPLOAD_FOLDERS['authorized'], exist_ok=True)
os.makedirs(UPLOAD_FOLDERS['blacklisted'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and processing."""
    
    # 1. Check for file
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # 2. Get upload type (authorized or blacklisted)
    upload_type = request.form.get('upload_type', 'authorized')
    if upload_type not in UPLOAD_FOLDERS:
        flash('Invalid upload type')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        target_folder = UPLOAD_FOLDERS[upload_type]
        save_path = os.path.join(target_folder, filename)
        
        # 3. Save the file
        file.save(save_path)
        print(f"[INFO] File saved to {save_path}")

        # 4. Log the upload event (Task 1)
        log_upload_event(filename, upload_type)

        # 5. Extract OCR data (Task 2)
        ocr_result = extract_data_from_image(save_path)

        # 6. Update registered_plates.json (Task 3)
        if "error" not in ocr_result:
            update_registered_plates(ocr_result, upload_type)
        
        # 7. Render the page again, showing the result (Task 4)
        flash(f'File "{filename}" uploaded to "{upload_type}".')
        
        # Pass the ocr_result to the template
        return render_template('index.html', ocr_result=ocr_result)

    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg, bmp')
        return redirect(request.url)


if __name__ == '__main__':
    print("[INFO] Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, host='0.0.0.0', port=5000)