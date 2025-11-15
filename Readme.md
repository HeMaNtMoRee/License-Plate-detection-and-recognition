Step 1 — Create Virtual Environment (Python 3.10.0)
Make sure you have Python 3.10.0 installed on your system. Then run:

# Create a virtual environment named "venv" using Python 3.10.0
python3.10 -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

Step 2 — Install PaddlePaddle (CPU version)
Inside the activated virtual environment, run:
python -m pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

Step 3 — Install Other Required Libraries
pip install -r requirements.txt

Step 4 — Check Installed Versions
Make sure versions match your requirement:
write command in terminal: pip list

You should see lib versions:
tensorflow       2.13.0
paddleocr        3.1.0
opencv-python    4.12.0.88
numpy            1.24.3

If any version is different, install the correct one:

pip install tensorflow==2.13.0 opencv-python==4.12.0.88 paddleocr==2.6.0 numpy==1.24.3 --force-reinstall

Step 5 — Create Required Folder
In the same directory where tod_anpr.py is located, create the folder:
mkdir cropped_number_plate_images

Step 6 — Run the Script
python tod_anpr.py
