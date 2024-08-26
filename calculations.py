import cv2
import numpy as np
import statistics
from PIL import Image
import os
import logging
import io



def blob(image_stream, k, b, circ1, iner1, conv1, area_min, area_max):
    # Ensure the stream is at the start
    image_stream.seek(0)
    
    # Load the image from the stream into a PIL Image
    pil_image = Image.open(image_stream)
    
    # Convert the PIL image to a NumPy array (OpenCV format)
    cv_image = np.array(pil_image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (k, k), 0)  # GaussianBlur with kernel size 11x11
    
    # Apply Otsu's thresholding after Gaussian blurring
    ret, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the thresholded image to make white regions black and black regions white
    inverted_thresh = cv2.bitwise_not(thresh)
    
    
    
    # Find contours on the inverted image
    contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = float(area) / hull_area
            else:
                convexity = 0
            
            if circularity > 0.6 and convexity > 0.1:
                # Apply morphological closing to the selected contour
                mask = np.zeros_like(thresh)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                kernel = np.ones((3, 3), np.uint8)
                closed_region = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                thresh[closed_region == 255] = 0  # Set white regions to black in the original thresholded image
    
    # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = circ1  # Adjust as needed
    
    # Filter by inertia
    params.filterByInertia = True
    params.minInertiaRatio = iner1  # Adjust as needed
    
    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = conv1
    
    # Filter by area
    params.filterByArea = True
    params.minArea = area_min  # Adjust as needed
    params.maxArea = area_max  # Adjust as needed
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(thresh)
   

    return len(keypoints)


def AI_calculation(image):
    """Perform AI calculation on the provided image."""
    params = [7, 0, 0.8615717345511243, 0.2832872144801636, 0.9309510243279392, 96.81274727295487, 1041.0370485937078]
    try:
        image_stream = io.BytesIO()
        image.save(image_stream, format='PNG')
        image_stream.seek(0)
        value = blob(image_stream, *params)
        return max(0, min(value, 500))
    except Exception as e:
        logging.error(f"Error during AI calculation: {e}")
        return None



def process_image(image_stream):
    """Process a single image: crop and perform AI calculation."""
    try:
        image_stream.seek(0)  # Ensure the stream is at the start
        with Image.open(image_stream) as img:
            value = AI_calculation(img)  #MAde changes here
            return value
    except Exception as e:
        logging.error(f"Error processing in-memory image: {e}")
        return None

from ultralytics import YOLO


def inference(image, model):
    try:
        results = model.predict(image, imgsz=2464, conf=0.5)
        logging.info("Predicted result: "+str(len(results[0].boxes)))
        return len(results[0].boxes)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None