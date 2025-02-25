import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1) Configuration
model_path = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\GoogLeNet_augmented.pth"
class_names = ["fine", "mild", "severe"]  # Must match how you trained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Recreate GoogLeNet with the same architecture used during training
num_classes = 3
model = models.googlenet(weights=None, aux_logits=False, init_weights=False)

# Modify the classifier to match the trained model
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Move to device
model.to(device)

# 3) Load state_dict with safe weights-only mode
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=True)

model.eval()

# 4) Define the transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    """Classifies a single image and returns the predicted class label."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f"Error loading image: {e}"

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return class_names[predicted_idx.item()]

if __name__ == "__main__":
    image_path = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\testfine.png"
    prediction = classify_image(image_path)
    print(f"Prediction: {prediction}")
