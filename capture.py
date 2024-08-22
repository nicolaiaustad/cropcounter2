
from datetime import datetime
from picamera2 import Picamera2
import time
import io
import os
import logging
from PIL import Image



# def capture_image(picam2, counter, capture_images, longitude, latitude):
#     now = datetime.now()
#     timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
#     image_stream = io.BytesIO()
#     picam2.capture_file(image_stream, format='png')
#     image_stream.seek(0)  # Rewind the stream to the beginning
    
#     if not capture_images:
#         return image_stream, timestamp, (longitude, latitude)
    
#     # # Save the image to the running_images directory
#     # directory = "/home/nicolaiaustad/Desktop/CropCounter/running_images"
#     # if not os.path.exists(directory):
#     #     os.makedirs(directory)
#     # png_filename = f"{directory}/image_{timestamp}__lat{latitude}_lon{longitude}.png"
    
#     # with open(png_filename, 'wb') as f:
#     #     f.write(image_stream.getbuffer())
    
#     #logging.info(f"Captured image with timestamp {timestamp}, GPS: ({latitude}, {longitude})")
    
#     # Optionally save every 10th image as JPEG
#     if counter % 50 == 0:
#         jpeg_directory = "/home/nicolaiaustad/Desktop/CropCounter/logged_images"
#         if not os.path.exists(jpeg_directory):
#             os.makedirs(jpeg_directory)
#         jpeg_filename = f"{jpeg_directory}/bygg_image_{timestamp}_lat{latitude}_lon{longitude}.jpeg"
        
#         # Convert PNG to JPEG and save
#         image_stream.seek(0)
#         with Image.open(image_stream) as img:
#             img = img.convert("RGB")
#             img.save(jpeg_filename, "JPEG", quality=85)
        
#         logging.info(f"Saved JPEG {jpeg_filename}")
    
#     image_stream.seek(0)  # Rewind the stream again for further processing
#     return image_stream, timestamp, (longitude, latitude)


from PIL import Image
import os
import io
import logging
from datetime import datetime

def capture_image(picam2, counter, capture_images, longitude, latitude):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    image_stream = io.BytesIO()
    picam2.capture_file(image_stream, format='png')
    image_stream.seek(0)  # Rewind the stream to the beginning
    
    # Open the image from the stream
    image = Image.open(image_stream).convert("RGB")
    
    if not capture_images:
        return image, timestamp, (longitude, latitude)
    
    # Optionally save every 50th image as JPEG
    if counter % 50 == 0:
        jpeg_directory = "/home/nicolaiaustad/Desktop/CropCounter/logged_images"
        if not os.path.exists(jpeg_directory):
            os.makedirs(jpeg_directory)
        jpeg_filename = f"{jpeg_directory}/bygg_image_{timestamp}_lat{latitude}_lon{longitude}.jpeg"
        
        # Save the image as JPEG
        image.save(jpeg_filename, "JPEG", quality=85)
        
        logging.info(f"Saved JPEG {jpeg_filename}")
    
    return image, timestamp, (longitude, latitude)
