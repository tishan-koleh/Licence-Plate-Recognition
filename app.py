from flask import Flask, render_template, request, send_file
import cv2
import numpy as np

app = Flask(__name__)

# Load your image processing code from image_processing.py
from image_processing import process_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded image to a temporary location
        uploaded_image_path = 'temp_image.jpeg'
        file.save(uploaded_image_path)
        
        # Process the uploaded image using your image processing code
        processed_image = process_image(uploaded_image_path)
        
        # Save the processed image
        processed_image_path = 'temp_result_image.jpg'
        cv2.imwrite(processed_image_path, processed_image)
        
        # Return the processed image to the user
        return send_file(processed_image_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
