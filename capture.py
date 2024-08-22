
from datetime import datetime
from picamera2 import Picamera2
import time
import io
import os
import logging
from PIL import Image

# def capture_image(picam2, counter, capture_images, longitude, latitude):
#     #Add logic to extract gps coordinates for images taken
   
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     if not capture_images:
#         image_stream = io.BytesIO()
#         picam2.capture_file(image_stream, format='png')
#         # print("Captured image but not saving it.")
#         #logging.info("Captured image but not saving it.")
#     else:
#         directory = "/home/nicolaiaustad/Desktop/CropCounter/images/Folder_A"
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         filename = f"{directory}/image_{timestamp}__lat{latitude}_lon{longitude}.png"
#         picam2.capture_file(filename)
        
#         logging.info(f"Captured {filename}")
#         #print(f"Captured {filename}")
        
#     return latitude, longitude





def capture_image(picam2, counter, capture_images, longitude, latitude):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    image_stream = io.BytesIO()
    picam2.capture_file(image_stream, format='png')
    image_stream.seek(0)  # Rewind the stream to the beginning
    
    if not capture_images:
        return image_stream, timestamp, (longitude, latitude)
    
    # # Save the image to the running_images directory
    # directory = "/home/nicolaiaustad/Desktop/CropCounter/running_images"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # png_filename = f"{directory}/image_{timestamp}__lat{latitude}_lon{longitude}.png"
    
    # with open(png_filename, 'wb') as f:
    #     f.write(image_stream.getbuffer())
    
    #logging.info(f"Captured image with timestamp {timestamp}, GPS: ({latitude}, {longitude})")
    
    # Optionally save every 10th image as JPEG
    if counter % 50 == 0:
        jpeg_directory = "/home/nicolaiaustad/Desktop/CropCounter/logged_images"
        if not os.path.exists(jpeg_directory):
            os.makedirs(jpeg_directory)
        jpeg_filename = f"{jpeg_directory}/bygg_image_{timestamp}_lat{latitude}_lon{longitude}.jpeg"
        
        # Convert PNG to JPEG and save
        image_stream.seek(0)
        with Image.open(image_stream) as img:
            img = img.convert("RGB")
            img.save(jpeg_filename, "JPEG", quality=85)
        
        logging.info(f"Saved JPEG {jpeg_filename}")
    
    image_stream.seek(0)  # Rewind the stream again for further processing
    return image_stream, timestamp, (longitude, latitude)


#!/usr/bin/python3





# import logging
# import signal
# import sys
# from picamera2 import Picamera2
# #from picamera2.encoders import PNGEncoder
# from picamera2.outputs import FileOutput

# logging.basicConfig(level=logging.DEBUG)

# def signal_handler(sig, frame):
#     logging.info("Signal received, stopping camera...")
#     picam2.stop()
#     logging.info("Camera stopped.")
#     sys.exit(0)

# logging.info("Starting camera setup...")
# picam2 = Picamera2()

# # Configure for high-resolution still image capture
# picam2.configure(picam2.create_still_configuration(main={"size": (3280, 2464)}))#, 'format': 'RGB888'}))  # Adjust to the maximum resolution of your camera sensor
# #picam2.configure(picam2.create_still_configuration(main={"size": (1456, 1088)}))
# picam2.set_controls({
#     "AnalogueGain": 1,  # Adjust gain to reduce noise
#     "ExposureTime": 30000,  # Increase exposure time for better lighting
#     "Brightness": 0.4,  # Adjust brightness as needed
#     "Contrast": 2,  # Increase contrast for better distinction
#     "Saturation": 0.5,  # Increase saturation for more vibrant colors
#     "Sharpness": 1.8,  # Increase sharpness for better detail
#     #"awb_mode": "off",  # Auto white balance
#     "AeEnable": True  # Enable automatic exposure
# })

# signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)

# try:
#     logging.info("Starting camera...")
#     picam2.start()
#     logging.info("Camera started.")
    
#     # Capture a high-resolution image
#     logging.info("Capturing image...")
#     picam2.capture_file("image20.png", format="png")
#     logging.info("Image captured and saved as image.png.")

# except Exception as e:
#     logging.error("Error occurred: %s", e)
# finally:
#     logging.info("Stopping camera...")
#     picam2.stop()
#     logging.info("Camera stopped.")





# # from picamera2 import Picamera2

# # picam2 = Picamera2()
# # picam2.start()
# # picam2.capture_file("test_image.jpg")
# # picam2.stop()
# # picam2.close()

