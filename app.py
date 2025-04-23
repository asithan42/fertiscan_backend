
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import predict_image, initialize_model, load_model_weights, load_class_indices
from flasgger import Swagger, swag_from
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Folder paths
LEAF_FOLDER = 'rice_leaf_image'
RESULT_FOLDER = 'results'
UPLOADS_FOLDER = 'uploads'

# Create folders if they don't exist
for folder in [LEAF_FOLDER, RESULT_FOLDER, UPLOADS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Image processing utilities
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var() < threshold

def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def is_dark_or_bright(image, low_thresh=80, high_thresh=180):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return mean < low_thresh or mean > high_thresh

def apply_clahe_soft(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess_leaf_image(image):
    image = cv2.resize(image, (224, 224))
    processed = image.copy()

    if is_blurry(processed):
        print("Blurry image detected. Applying sharpening...")
        processed = sharpen(processed)

    if is_dark_or_bright(processed):
        print("Brightness/contrast issue detected. Applying soft CLAHE...")
        processed = apply_clahe_soft(processed)

    return processed

@app.route('/upload-leaf', methods=['POST'])
@swag_from({
    'tags': ['Rice Leaf'],
    'summary': 'Upload rice leaf image for preprocessing and prediction',
    'description': 'Upload a rice leaf image. The image will be preprocessed and classified.',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Rice leaf image file'
        }
    ],
    'responses': {
        200: {'description': 'Image processed successfully'},
        400: {'description': 'Bad request'},
        500: {'description': 'Server error'}
    }
})
def upload_leaf_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filename = image.filename
    leaf_path = os.path.join(LEAF_FOLDER, filename)
    image.save(leaf_path)

    processed_filename = 'processed_leaf_image.jpg'
    processed_path = os.path.join(RESULT_FOLDER, processed_filename)

    try:
        img_leaf = cv2.imread(leaf_path)
        if img_leaf is None:
            return jsonify({'error': 'Could not read uploaded image'}), 500

        # Preprocess the leaf image
        processed_leaf = preprocess_leaf_image(img_leaf)
        cv2.imwrite(processed_path, processed_leaf)

        # Load model and class indices
        class_indices = load_class_indices('model/classes_indices.json')
        num_classes = len(class_indices)
        model = initialize_model(num_classes)
        model = load_model_weights(model)

        # Predict
        predicted_class = predict_image(processed_path, model, class_indices)

        return jsonify({
            'message': 'Leaf image uploaded, processed, and predicted successfully',
            'original': filename,
            'processed': processed_filename,
            'predicted_class': predicted_class
        }), 200

    except Exception as e:
        return jsonify({'error': f'{str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
