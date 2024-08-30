import os
import time
from datetime import datetime
from picamera2 import Picamera2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the camera
picam2 = Picamera2()

# Create a camera configuration
camera_config = picam2.create_still_configuration(
    main={
        "size": (2464, 2464),  # Maximum resolution for the camera
        "format": "RGB888"  # Use a high-quality format
    },
    controls={
        "AnalogueGain": 3,
        "ExposureTime": 1000,
        "Brightness": 0.0,
        "Contrast": 1,
        "Saturation": 1,
        "Sharpness": 4,
        "AeEnable": False,
        "AwbEnable": True
    }
)

# Configure the camera with the specified settings
picam2.configure(camera_config)

# Start the camera
picam2.start()
time.sleep(2)  # Allow the camera to warm up

# Directory to save images
image_dir = "captured_images"
os.makedirs(image_dir, exist_ok=True)

# Capture and save image
try:
    # Create a timestamped filename
   
    counter = 0
    while counter < 10:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(image_dir, f"training_slow1000_{timestamp}.png")
        image_path2 = os.path.join(image_dir, f"training_slow1000_{timestamp}.jpeg")
        
        # Capture the image
        picam2.capture_file(image_path)
        picam2.capture_file(image_path2)
        
        counter += 1
        time.sleep(2)
        logger.info(f"Image saved at {image_path} and {image_path2}")

finally:
    # Stop the camera
    picam2.stop()
