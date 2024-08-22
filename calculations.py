import cv2
import numpy as np
import statistics
from PIL import Image
import os
import logging
import io

# ### Blob detection method
# def blob(image, k, b, circ1, circ2, iner1, iner2, conv1, conv2, area_min, area_max):
#     # Convert the image to grayscale
#     gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

#     blurred_image = cv2.GaussianBlur(gray, (k, k), 0)  #Apply gaussian blur to the image
    
#     thresh = cv2.adaptiveThreshold(blurred_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,b,2) #Apply adaptive gaussian thresholding to the image
    
#     # Set up the SimpleBlobDetector parameters
#     params = cv2.SimpleBlobDetector_Params()

#     # Filter by circularity
#     params.filterByCircularity = True
#     params.minCircularity = circ1 # Adjust as needed
#     params.maxCircularity = circ2

#     # Filter by inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = iner1  # Adjust as needed
#     params.maxInertiaRatio = iner2
    
#     # Filter by Convexity
#     params.filterByConvexity = True
#     params.minConvexity = conv1
#     params.maxConvexity = conv2

#     # Filter by area
#     params.filterByArea = True
#     params.minArea = area_min # Adjust as needed
#     params.maxArea = area_max  # Adjust as needed
    
#     # Create a detector with the parameters
#     detector = cv2.SimpleBlobDetector_create(params)

#     # Detect blobs
#     keypoints = detector.detect(thresh)
   

#     return len(keypoints)

def blob(image_stream, k, b, circ1, iner1, conv1, area_min, area_max):
    # Ensure the stream is at the start
    image_stream.seek(0)
    
    # Load the image from the stream into a PIL Image
    pil_image = Image.open(image_stream)
    
    # Convert the PIL image to a NumPy array (OpenCV format)
    cv_image = np.array(pil_image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

    # # Apply Gaussian blur to the image
    # blurred_image = cv2.GaussianBlur(gray, (k, k), 0)
    
    # # Apply adaptive Gaussian thresholding to the image
    # thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, 3)
    
    # # Invert the thresholded image to make white regions black and black regions white
    # inverted_thresh = cv2.bitwise_not(thresh)
    
    
    
    
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

    
# def AI_calculation(image_path):
#     #Best parameters are stored privately and not shared on GitHub.
#     params= [7, 7, 0.7389686996692504, 1, 0.37438265817010036, 1, 0.526007653770587, 1, 42.77016806408365, 483.1027394129678]
#     #Dummy function for
#     value = blob(image_path, 7, 7, 0.7389686996692504, 1, 0.37438265817010036, 1, 0.526007653770587, 1, 42.77016806408365, 483.1027394129678)
#     return value

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
            # Crop the image (example crop: 240 pixels from left and right)
            #cropped_img = img.crop((0, 0, img.width, img.height))
            # Perform AI calculation
            value = AI_calculation(img)  #MAde changes here
            return value
    except Exception as e:
        logging.error(f"Error processing in-memory image: {e}")
        return None
    
def inference(image, model):
    try:
        # Run inference
        results = model.predict(image)
        
        return len(results[0].boxes)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None
