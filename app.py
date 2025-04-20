
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from model_utils import predict_image
# from flasgger import Swagger, swag_from
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import models
# import json

# # Import correction pipeline
# from correction_pipeline import (
#     extract_reference_card,
#     detect_reference_colors,
#     calculate_color_transform,
#     segment_leaf,
#     apply_color_correction_with_mask
# )

# app = Flask(__name__)
# CORS(app)
# swagger = Swagger(app)

# # Folder paths
# LCC_FOLDER = 'lcc_chart'
# LEAF_FOLDER = 'rice_leaf_image'
# RESULT_FOLDER = 'results'
# UPLOADS_FOLDER = 'uploads'  # Optional

# # Create folders if they don't exist
# for folder in [LCC_FOLDER, LEAF_FOLDER, RESULT_FOLDER, UPLOADS_FOLDER]:
#     os.makedirs(folder, exist_ok=True)


# # Load class index dictionary
# def load_class_indices(class_indices_path='model/classes_indices.json'):
#     """
#     Loads the class index to class label mapping from a JSON file.
#     """
#     if not os.path.exists(class_indices_path):
#         raise FileNotFoundError(f"{class_indices_path} does not exist.")
    
#     with open(class_indices_path, 'r') as f:
#         return json.load(f)


# # Initialize resnet18 architecture
# def initialize_model(num_classes):
#     """
#     Initializes a resnet18 model, replacing the final fully connected layer 
#     to match the number of classes.
#     """
#     model = models.resnet18(pretrained=False)  # Load ResNet18 without pretrained weights
#     model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust output layer to match the number of classes
#     return model


# # Load saved weights
# def load_model_weights(model, model_weights_path='model/rice_leaf_model.pth'):
#     """
#     Loads the model weights from the specified file.
#     """
#     if not os.path.exists(model_weights_path):
#         raise FileNotFoundError(f"{model_weights_path} does not exist.")
    
#     model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
#     return model


# @app.route('/upload-lcc', methods=['POST'])
# @swag_from({
#     'tags': ['LCC Chart'],
#     'summary': 'Upload LCC chart image',
#     'description': 'Upload a single image file of the LCC chart.',
#     'consumes': ['multipart/form-data'],
#     'parameters': [
#         {
#             'name': 'image',
#             'in': 'formData',
#             'type': 'file',
#             'required': True,
#             'description': 'LCC chart image file'
#         }
#     ],
#     'responses': {
#         200: {
#             'description': 'LCC chart uploaded successfully',
#             'examples': {
#                 'application/json': {
#                     'message': 'LCC chart uploaded successfully',
#                     'filename': 'LCC_chart_image.jpg'
#                 }
#             }
#         },
#         400: {
#             'description': 'Bad request (missing file)'
#         }
#     }
# })
# def upload_lcc_chart():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part in request'}), 400

#     image = request.files['image']
#     if image.filename == '':
#         return jsonify({'error': 'No image selected'}), 400

#     filename = 'LCC_chart_image.jpg'
#     filepath = os.path.join(LCC_FOLDER, filename)
#     image.save(filepath)

#     return jsonify({'message': 'LCC chart uploaded successfully', 'filename': filename}), 200


# @app.route('/upload-leaf', methods=['POST'])
# @swag_from({
#     'tags': ['Rice Leaf'],
#     'summary': 'Upload rice leaf image for correction',
#     'description': 'Upload a rice leaf image. The image will be corrected (brightness enhanced) and saved.',
#     'consumes': ['multipart/form-data'],
#     'parameters': [
#         {
#             'name': 'image',
#             'in': 'formData',
#             'type': 'file',
#             'required': True,
#             'description': 'Rice leaf image file'
#         }
#     ],
#     'responses': {
#         200: {
#             'description': 'Image uploaded and corrected',
#             'examples': {
#                 'application/json': {
#                     'message': 'Leaf image uploaded and corrected successfully',
#                     'original': 'leaf_image.jpg',
#                     'corrected': 'corrected_leaf_image.jpg',
#                     'predicted_class': 'swap3'
#                 }
#             }
#         },
#         400: {
#             'description': 'Bad request (missing file)'
#         },
#         500: {
#             'description': 'Server error (image read failed or processing error)'
#         }
#     }
# })
# def upload_leaf_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part in request'}), 400

#     image = request.files['image']
#     if image.filename == '':
#         return jsonify({'error': 'No image selected'}), 400

#     filename = image.filename
#     leaf_path = os.path.join(LEAF_FOLDER, filename)
#     image.save(leaf_path)

#     corrected_filename = 'corrected_leaf_image.jpg'
#     corrected_path = os.path.join(RESULT_FOLDER, corrected_filename)

#     try:
#         img_leaf = cv2.imread(leaf_path)
#         if img_leaf is None:
#             return jsonify({'error': 'Could not read uploaded image'}), 500

#         # Load LCC chart image
#         lcc_chart_path = os.path.join(LCC_FOLDER, 'LCC_chart_image.jpg')
#         if not os.path.exists(lcc_chart_path):
#             return jsonify({'error': 'LCC chart image not found. Please upload it first.'}), 500

#         ref_card = extract_reference_card(lcc_chart_path)
#         detected_colors = detect_reference_colors(ref_card)

#         # Sample known reference HSV colors for LCC patches (example values)
#         known_colors = np.array([
#             [60, 100, 100],   # Patch 1
#             [45, 150, 130],   # Patch 2
#             [30, 200, 160],   # Patch 3
#             [15, 250, 190]    # Patch 4
#         ], dtype='float32')

#         transform_matrix = calculate_color_transform(detected_colors, known_colors)
#         leaf_mask = segment_leaf(img_leaf)
#         corrected_leaf = apply_color_correction_with_mask(img_leaf, transform_matrix, leaf_mask)

#         cv2.imwrite(corrected_path, corrected_leaf)

#         # ✅ Load the model and class indices
#         class_indices = load_class_indices('model/classes_indices.json')
#         num_classes = len(class_indices)
#         model = initialize_model(num_classes)
#         model = load_model_weights(model)

#         # ✅ Predict using corrected image
#         predicted_class = predict_image(corrected_path, model, class_indices)

#         return jsonify({
#             'message': 'Leaf image uploaded, corrected, and predicted successfully',
#             'original': filename,
#             'corrected': corrected_filename,
#             'predicted_class': predicted_class
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import predict_image, initialize_model, load_model_weights, load_class_indices
from flasgger import Swagger, swag_from
import os
import cv2
import numpy as np

# Import color correction pipeline
from correction_pipeline import (
    extract_reference_card,
    detect_reference_colors,
    calculate_color_transform,
    segment_leaf,
    apply_color_correction_with_mask
)

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Folder paths
LCC_FOLDER = 'lcc_chart'
LEAF_FOLDER = 'rice_leaf_image'
RESULT_FOLDER = 'results'
UPLOADS_FOLDER = 'uploads'

# Create folders if they don't exist
for folder in [LCC_FOLDER, LEAF_FOLDER, RESULT_FOLDER, UPLOADS_FOLDER]:
    os.makedirs(folder, exist_ok=True)


@app.route('/upload-lcc', methods=['POST'])
@swag_from({
    'tags': ['LCC Chart'],
    'summary': 'Upload LCC chart image',
    'description': 'Upload a single image file of the LCC chart.',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'LCC chart image file'
        }
    ],
    'responses': {
        200: {'description': 'LCC chart uploaded successfully'},
        400: {'description': 'Bad request'}
    }
})
def upload_lcc_chart():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filename = 'LCC_chart_image.jpg'
    filepath = os.path.join(LCC_FOLDER, filename)
    image.save(filepath)

    return jsonify({'message': 'LCC chart uploaded successfully', 'filename': filename}), 200


@app.route('/upload-leaf', methods=['POST'])
@swag_from({
    'tags': ['Rice Leaf'],
    'summary': 'Upload rice leaf image for correction and prediction',
    'description': 'Upload a rice leaf image. The image will be corrected using the LCC chart and classified.',
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

    corrected_filename = 'corrected_leaf_image.jpg'
    corrected_path = os.path.join(RESULT_FOLDER, corrected_filename)

    try:
        img_leaf = cv2.imread(leaf_path)
        if img_leaf is None:
            return jsonify({'error': 'Could not read uploaded image'}), 500

        # Load LCC chart image
        lcc_chart_path = os.path.join(LCC_FOLDER, 'LCC_chart_image.jpg')
        if not os.path.exists(lcc_chart_path):
            return jsonify({'error': 'LCC chart image not found. Please upload it first.'}), 500

        ref_card = extract_reference_card(lcc_chart_path)
        detected_colors = detect_reference_colors(ref_card)

        # Known HSV reference values (example only - you should update based on your LCC chart)
        known_colors = np.array([
            [119, 136, 40],
            [482, 98, 33],
            [83, 98, 59],
            [54, 63, 44]
        ], dtype='float32')

        transform_matrix = calculate_color_transform(detected_colors, known_colors)
        leaf_mask = segment_leaf(img_leaf)
        corrected_leaf = apply_color_correction_with_mask(img_leaf, transform_matrix, leaf_mask)

        cv2.imwrite(corrected_path, corrected_leaf)

        # Load model and class indices
        class_indices = load_class_indices('model/classes_indices.json')
        num_classes = len(class_indices)
        model = initialize_model(num_classes)
        model = load_model_weights(model)

        # Predict
        predicted_class = predict_image(corrected_path, model, class_indices)

        return jsonify({
            'message': 'Leaf image uploaded, corrected, and predicted successfully',
            'original': filename,
            'corrected': corrected_filename,
            'predicted_class': predicted_class
        }), 200

    except Exception as e:
        return jsonify({'error': f'{str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

