from datetime import datetime
from picamera2 import Picamera2
import time
from PIL import Image
import io
import calculations
import random
import pyproj
import maps
import numpy as np
import pandas as pd
import capture
import os
import load_settings
import logging
from inotify.adapters import Inotify
import signal
import gps_func
from multiprocessing import Pool, cpu_count, TimeoutError, active_children
import sys
import select
import torch
from ultralytics import YOLO
from collections import deque

# Setting up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Adjust the logging level for Picamera2
picamera2_logger = logging.getLogger('picamera2')
picamera2_logger.setLevel(logging.WARNING)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('/home/nicolaiaustad/Desktop/CropCounter/run.log')
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
# Logging to check performance through script
logging.basicConfig(
    filename='/home/nicolaiaustad/Desktop/CropCounter/run.log',
    level=logging.DEBUG,  # Adjust as needed
    format='%(asctime)s %(levelname)s %(message)s'
)



# Signal handler to catch termination signals
def signal_handler(sig, frame):
    
    logging.info("Signal received, exiting...")
    raise KeyboardInterrupt 
    


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)



def process_remaining_results(results):
    while results:
        for timestamp, result in results:
            if result.ready():
                try:
                    print(len(results))
                except TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Error processing image with timestamp {timestamp}: {e}")
        
        results = [(timestamp, result) for timestamp, result in results if not result.ready()]

def collect_result(result):
    # Placeholder function to handle the result
    logging.info(f"Processed result: {result}")
    
    
def main(capture_images=True, num_cores=4):
    logging.info(f'run.py script started with {num_cores} cores')
    mount_point = '/mnt/usb_settings'
    device = '/dev/sda1'  # Adjust this as needed
    mount_point = load_settings.mount_usb(mount_point, device)  # Ensure the USB is mounted
    settings_dest = '/tmp/settings.txt'  # Destinations defined on the USB stick
    shapefiles_dest = '/tmp/shapefiles'
    load_settings.copy_settings_and_shapefiles(mount_point, settings_dest, shapefiles_dest)  # Copy settings and shapefiles from USB
    settings = load_settings.read_settings_file(settings_dest)  # Read settings

    with open('/tmp/usb_settings.env', 'w') as file:
        for key, value in settings.items():
            file.write(f'{key}={value}\n')
           
    job_name = str(settings['JOB_NAME'])
    grid_size = int(settings['GRID_SIZE'])
    float_number = float(settings['FLOAT'])
    print(job_name, grid_size, float_number)
    
    shapefile_path = load_settings.find_shapefile(shapefiles_dest)
    if not shapefile_path:
        print("No shapefile found in", shapefiles_dest)
        return
    print("Using shapefile:", shapefile_path)
    
    picam2 = Picamera2()
    
    camera_config = picam2.create_still_configuration(
        main={
            "size": (2464, 2464),  # Maximum resolution for the camera
            # "size": (640, 640),  # Maximum resolution for the camera
            
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

    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # Allow the camera to warm up

    grid_points, grid_gps, df_utm, df_gps, bound = maps.shp_to_grid(shapefile_path, grid_size)
    init_lon = grid_gps[0,0]
    init_lat = grid_gps[0,1]
    zone_number, hemisphere = maps.get_utm_zone(init_lon, init_lat)
    utm_crs = maps.create_utm_proj(zone_number, hemisphere)
    
    pool = Pool(processes=1)  # Use the specified number of CPU cores
    results = []
    image_metadata = {}  # Dictionary to store image metadata
    
    total_images_processed = 0
    processing_start_time = time.time()
    speed_store = deque(maxlen=5)
    
    # Load the YOLO model
    
    ncnn_model = YOLO("best_ncnn_model", task="detect")
   
    try:
        counter = 0
        logging.info('Now the While loop starts...')
        while True:
            start_time = time.time()
            longitude, latitude, satellites, speed = gps_func.get_gps()
            speed_store.append(speed)
            image_stream, timestamp, metadata = capture.capture_image(picam2, counter, True, longitude, latitude)
           
            image_metadata[timestamp] = metadata
            
            try:
                
                value = calculations.inference(image_stream, ncnn_model)
                if value is not None:
                    longitude, latitude = image_metadata[timestamp]
                    utm_x, utm_y = maps.transform_to_utm(longitude, latitude, utm_crs)
                    df_row = maps.find_grid_cell(utm_x, utm_y, grid_size, df_utm)
                    
                    if df_row is not None:
                        if counter % 5 == 0:
                            logging.info("Good signal. Satellite: "+str(satellites)+" Latitude: " + str(latitude)+ ", Longitude: " + str(longitude))
                        
                        if df_utm.at[df_row, "measured"] == False:
                            df_utm.at[df_row, "measured"] = True
                            df_utm.at[df_row, "values"] = int(value)
                        elif df_utm.at[df_row, "measured"] == True:
                            continue
                        else:
                            continue
                    else:
                        if counter % 5 == 0:
                            logging.info("GPS coordinate outside grid. Satellite: "+str(satellites)+" Latitude: " + str(latitude)+ ", Longitude: " + str(longitude))
                    
                    total_images_processed += 1
                del image_metadata[timestamp]
                
            except Exception as e:
                logging.error(f"Error processing image with timestamp {timestamp}: {e}")
            
            
            if max(speed_store) < 0.5:
                try:
                    maps.make_heatmap_and_save(df_utm, grid_size, f'/tmp/{job_name}_{counter}.png', f'/tmp/{job_name}_{counter}', utm_crs)
                except Exception as e:
                    logging.error(f"Error saving heatmap checkpoint: {e}")
                    
          
            # Update counter and timing
            counter += 1
            end_time = time.time()
            iteration_time = end_time - start_time
            
            # Calculate and log the images processed per second
            processing_end_time = time.time()
            elapsed_time = processing_end_time - processing_start_time
            if elapsed_time > 0:
                images_per_second = total_images_processed / elapsed_time
                logging.info(f"Total processed images: {total_images_processed:.2f}, in Time: {elapsed_time:.2f}")
            
            # Maintain a 2.5-second loop cycle
            time.sleep(max(0, 2.5 - iteration_time))
            
    except KeyboardInterrupt:
        print("Program interrupted")
        logging.info("Program interrupted")
       
    finally:
        try:
            picam2.stop()
        except Exception as e:
            logging.error(f"Error stopping camera: {e}")
       
        logging.info("Camera stopped and resources cleaned up")
        logging.info("Cleaning up resources...")
        

        try:
            maps.make_heatmap_and_save(df_utm, grid_size, f'/tmp/{job_name}_custom.png', f'/tmp/{job_name}_custom', utm_crs) #Creates custom smoothed heatmap
            maps.generate_idw_heatmap(df_utm, bound, grid_size, f'/tmp/{job_name}_idw.png', f'/tmp/{job_name}_idw', utm_crs) #Creates IDW heatmap
            
            heatmap_folder = os.path.join(mount_point, 'generated_heatmaps')
            generated_shapefiles_folder = os.path.join(mount_point, 'generated_shapefiles')
            os.makedirs(heatmap_folder, exist_ok=True)
            os.makedirs(generated_shapefiles_folder, exist_ok=True)
            load_settings.copy_files(f'/tmp/{job_name}_custom.png', heatmap_folder)
            load_settings.copy_files(f'/tmp/{job_name}_custom', generated_shapefiles_folder)
            
            load_settings.copy_files(f'/tmp/{job_name}_idw.png', heatmap_folder)
            load_settings.copy_files(f'/tmp/{job_name}_idw', generated_shapefiles_folder)
            
            time.sleep(4)
            load_settings.unmount_usb(mount_point)
            logging.info('Now generated files should be saved...')
        except Exception as e:
            logging.error(f"Error during final file handling: {e}")
            
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()
            
if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Re-running the script with sudo...")
        logging.info("Re-running the script with sudo...")
        try:
            load_settings.subprocess.run(['sudo', 'python3'] + load_settings.sys.argv)
          
            
        except KeyboardInterrupt:
            pass
    else:
      
        
        main(capture_images=True, num_cores=4)
