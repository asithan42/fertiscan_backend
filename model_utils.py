
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
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1024, 16, bias=False),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.Linear(16, num_classes)
    )
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
    transforms.Resize((100, 100)),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_label = list(class_indices.keys())[list(class_indices.values()).index(predicted.item())]
    return class_label
