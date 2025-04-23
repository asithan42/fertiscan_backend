
import cv2
import numpy as np

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
        processed = sharpen(processed)

    if is_dark_or_bright(processed):
        processed = apply_clahe_soft(processed)

    return processed
