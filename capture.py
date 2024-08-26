
from datetime import datetime
from picamera2 import Picamera2
import time
import io
import os
import logging
from PIL import Image

def capture_image(picam2, counter, capture_images, longitude, latitude):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    image_stream = io.BytesIO()
    picam2.capture_file(image_stream, format='png')
    image_stream.seek(0)  # Rewind the stream to the beginning
    
    # Open the image from the stream
    image_temp = Image.open(image_stream).convert("RGB")
    image = image_temp.copy()
    
    
    if not capture_images:
        return image, timestamp, (longitude, latitude)
    
    # Optionally save every 50th image as JPEG
    if counter % 30 == 0:
        jpeg_directory = "/home/nicolaiaustad/Desktop/CropCounter/logged_images"
        if not os.path.exists(jpeg_directory):
            os.makedirs(jpeg_directory)
        jpeg_filename = f"{jpeg_directory}/bygg_raset_image_{timestamp}_lat{latitude}_lon{longitude}.jpeg"
        
        # Save the image as JPEG
        image_temp.save(jpeg_filename, "JPEG", quality=85)
        
        logging.info(f"Saved JPEG {jpeg_filename}")
    
    elif counter % 201 == 0:
        jpeg_directory = "/home/nicolaiaustad/Desktop/CropCounter/logged_images"
        if not os.path.exists(jpeg_directory):
            os.makedirs(jpeg_directory)
        jpeg_filename = f"{jpeg_directory}/PNG_bygg_raset_image_{timestamp}_lat{latitude}_lon{longitude}.png"
        
        # Save the image as JPEG
        image_temp.save(jpeg_filename, "png")
        
        logging.info(f"Saved png {jpeg_filename}")
    
    return image, timestamp, (longitude, latitude)

