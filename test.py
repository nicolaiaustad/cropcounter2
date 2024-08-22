# import io
# from picamera2 import Picamera2
# from PIL import Image
# from ultralytics import YOLO
# import time
# from multiprocessing import Pool

# # Load the YOLO model
# model = YOLO('/home/nicolaiaustad/Desktop/CropCounter2/model/best.pt')

# def capture_image_in_memory(picam2):
#     """Capture an image from the camera and store it in memory."""
#     image_stream = io.BytesIO()
#     picam2.capture_file(image_stream, format='png')
#     image_stream.seek(0)  # Rewind the stream to the beginning
#     image = Image.open(image_stream).convert("RGB")
#     return image

# def split_image(image, num_patches):
#     """Split the image into a specified number of patches."""
#     width, height = image.size
#     patch_width = width // 2
#     patch_height = height // 2
    
#     patches = []
#     for i in range(2):
#         for j in range(2):
#             box = (i * patch_width, j * patch_height, (i + 1) * patch_width, (j + 1) * patch_height)
#             patches.append(image.crop(box))
    
#     return patches

# def run_inference_on_patch(patch):
#     """Run inference on a single patch."""
#     results = model.predict(patch, imgsz= 1232)
#     return results

# def process_image(image):
#     """Split an image into patches and run inference on each patch using multiprocessing."""
#     patches = split_image(image, num_patches=4)

#     with Pool(processes=4) as pool:
#         results = pool.map(run_inference_on_patch, patches)

#     return results

# if __name__ == "__main__":
#     # Initialize and start the camera
#     picam2 = Picamera2()
#     camera_config = picam2.create_still_configuration(
#         main={
#             "size": (2464, 2464),  # Adjust size if needed
#             "format": "RGB888"
#         }
#     )
#     picam2.configure(camera_config)
#     picam2.start()

#     # Allow camera to warm up
#     time.sleep(2)

#     try:
#         while True:
#             # Capture image in memory
#             image = capture_image_in_memory(picam2)

#             # Process the image by splitting it into patches and running inference
#             results = process_image(image)

#             # Print the results for each patch
#             for i, result in enumerate(results):
#                 print(f"Patch {i + 1}: {len(result[0].boxes)} boxes detected")

#             # Wait before capturing the next image
#             time.sleep(2)  # Adjust timing as needed
#     finally:
#         picam2.stop()  # Stop the camera when done

import os
import time
from PIL import Image
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
import logging

ncnn_model = YOLO("best_ncnn_model")

# Directory containing images
image_dir = '/home/nicolaiaustad/Desktop/CropCounter2/running_images'

# Filter and get all image files in the directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

def run_single_inference(ncnn_model, image_files):
    """Run inference on images one by one and measure time."""
    start_time = time.time()
    
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        results = ncnn_model.predict(image, imgsz=2816)
        print(f"Processed {image_file}: {len(results[0].boxes)} boxes detected")
    
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

def process_image_batch(ncnn_model, batch_files):
    """Process a batch of images and return the results."""
    results = []
    for image_file in batch_files:
        image = Image.open(image_file).convert("RGB")
        result = ncnn_model.predict(image, imgsz=2816)
        results.append((image_file, result))
    return results

def run_batch_inference(ncnn_model, image_files, batch_size=6):
    """Run inference on images in smaller batches and measure time."""
    start_time = time.time()
    pool = Pool(processes=cpu_count())
    results = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        result = pool.apply_async(process_image_batch, (ncnn_model, batch_files))
        results.append(result)
    
    pool.close()
    pool.join()

    for result in results:
        batch_results = result.get()
        for image_file, res in batch_results:
            print(f"Processed {image_file}: {len(res[0].boxes)} boxes detected")
    
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

if __name__ == "__main__":
    # Ensure there are images to process
    if not image_files:
        print("No images found in the directory. Please add images to the folder and try again.")
    else:
        # Run batch inference timing
        batch_inference_time = run_batch_inference(ncnn_model, image_files)
        print(f"Total time for batch inference: {batch_inference_time:.2f} seconds")
        
        # Run single inference timing
        single_inference_time = run_single_inference(ncnn_model, image_files)
        print(f"Total time for single image inference: {single_inference_time:.2f} seconds")
        
        # Compare the results
        print("\nComparison:")
        print(f"Single image inference took {single_inference_time:.2f} seconds")
