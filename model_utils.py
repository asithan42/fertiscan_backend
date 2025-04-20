# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import json
# import os

# # Load class index dictionary
# def load_class_indices(class_indices_path='model/classes_indices.json'):
#     """
#     Loads the class index to class label mapping from a JSON file.
#     """
#     if not os.path.exists(class_indices_path):
#         raise FileNotFoundError(f"{class_indices_path} does not exist.")
    
#     with open(class_indices_path, 'r') as f:
#         return json.load(f)

# # Define number of classes
# class_indices = load_class_indices('model/classes_indices.json')
# num_classes = len(class_indices)

# # Initialize resnet18 architecture
# def initialize_model(num_classes):
#     """
#     Initializes a resnet18 model, replacing the final fully connected layer 
#     to match the number of classes.
#     """
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

# # Load saved weights
# def load_model_weights(model, model_weights_path='model/rice_leaf_model.pth'):
#     """
#     Loads the model weights from the specified file using strict=False to avoid key mismatch errors.
#     """
#     if not os.path.exists(model_weights_path):
#         raise FileNotFoundError(f"{model_weights_path} does not exist.")
    
#     # Load weights with strict=False to allow for flexible loading
#     model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')), strict=False)
#     return model

# # Image transformation and prediction function
# def predict_image(image_path, model, class_indices):
#     """
#     Predict the class label of a given image using the trained model.
#     """
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The image at {image_path} was not found.")
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0)
    
#     model.eval()
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         _, predicted = torch.max(outputs, 1)

#     # Get class label from index
#     class_label = list(class_indices.keys())[list(class_indices.values()).index(predicted.item())]
#     return class_label

# # Main code to load the model, perform prediction and print results
# def main(image_path):
#     try:
#         model = initialize_model(num_classes)
#         model = load_model_weights(model)  # Load weights into the model
#         prediction = predict_image(image_path, model, class_indices)
#         print(f"Predicted class: {prediction}")
#     except Exception as e:
#         print(f"Error: {e}")

# # Example usage
# if __name__ == "__main__":
#     # Replace this with your actual image path during testing
#     main('corrected_leaf_image.jpg')


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

def load_class_indices(class_indices_path='model/classes_indices.json'):
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"{class_indices_path} does not exist.")
    
    with open(class_indices_path, 'r') as f:
        return json.load(f)

def initialize_model(num_classes):
    from torchvision.models import densenet121
    model = densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def load_model_weights(model, model_weights_path='model/rice_leaf_model.pth'):
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"{model_weights_path} does not exist.")
    
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')), strict=False)
    return model

def predict_image(image_path, model, class_indices):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image at {image_path} was not found.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_label = list(class_indices.keys())[list(class_indices.values()).index(predicted.item())]
    return class_label
