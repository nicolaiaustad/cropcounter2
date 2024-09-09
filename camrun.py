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
        "ExposureTime": 500,
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
    while counter < 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(image_dir, f"etraining_{timestamp}.png")
        image_path2 = os.path.join(image_dir, f"etraining_{timestamp}.jpeg")
        
        # Measure time before capturing
        start_time = time.time()
        
        # Capture the PNG image
        picam2.capture_file(image_path)
        
        # Measure time taken for PNG capture
        png_time = time.time() - start_time
        logger.info(f"PNG image saved at {image_path}. Time taken: {png_time:.2f} seconds")
        
        # Measure time before capturing JPEG
        start_time = time.time()
        
        # Capture the JPEG image
        picam2.capture_file(image_path2)
        
        # Measure time taken for JPEG capture
        jpeg_time = time.time() - start_time
        logger.info(f"JPEG image saved at {image_path2}. Time taken: {jpeg_time:.2f} seconds")
        
        counter += 1
        time.sleep(2)

finally:
    # Stop the camera
    picam2.stop()
