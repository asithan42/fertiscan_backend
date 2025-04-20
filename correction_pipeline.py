# correction_pipeline.py
 
import cv2
import numpy as np
import os
 
def extract_reference_card(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y+h, x:x+w]
 
def detect_reference_colors(reference_card, num_patches=4):
    hsv_card = cv2.cvtColor(reference_card, cv2.COLOR_BGR2HSV)
    h, w, _ = reference_card.shape
    patch_height = h // num_patches
    colors = [
        np.mean(hsv_card[i * patch_height:(i + 1) * patch_height, :].reshape(-1, 3), axis=0)
        for i in range(num_patches)
    ]
    return np.array(colors)
 
def calculate_color_transform(detected_colors, known_colors):
    A = np.array(detected_colors, dtype='float32')
    B = np.array(known_colors, dtype='float32')
    transform_matrix, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return transform_matrix
 
def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
 
def apply_color_correction_with_mask(image, transform_matrix, mask):
    corrected_image = image.astype('float32')
    h, w, c = corrected_image.shape
    reshaped = corrected_image.reshape(-1, 3)
    transformed = reshaped @ transform_matrix.T
    transformed = np.clip(transformed, 0, 255).astype('uint8').reshape(h, w, c)
    mask_3ch = cv2.merge([mask, mask, mask])
    output_image = np.where(mask_3ch == 255, transformed, image)
    return output_image