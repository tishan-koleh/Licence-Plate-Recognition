import cv2
import numpy as np
import easyocr
import imutils

def process_image(image_path):
    # Read the input image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction using bilateral filtering
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection using Canny
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Find contours and locate the text region
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    # Create a mask based on the text region location
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    # Crop the text region
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    
    # Perform text recognition using EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    # Extract the recognized text
    text = result[0][-2]

    if result:
    # Check if the first result has at least 3 elements
       if len(result[0]) >= 3:
        text = result[0][-2]
       else:
        text = "No text found"
    else:
        text = "No text found"

    if text == "No text found":
      return []    
    
    # Draw the recognized text on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60),
                      fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    
    return res  # Return the processed image as a NumPy array
