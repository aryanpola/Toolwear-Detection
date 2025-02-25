import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 1. Dataset Preparation

train_dir = "dataset/train"  # Contains fine_train_augmented, mild_train_augmented, severe_train_augmented
test_dir  = "dataset/test"   # Contains fine_test_augmented, mild_test_augmented, severe_test_augmented

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset  = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4)

num_classes = 3


# 2. Define ResNet-18 Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10


# 3. Training Loop

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# 4. Final Accuracy (Train)

model.eval()
correct_train, total_train = 0, 0
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
final_train_acc = 100.0 * correct_train / total_train


# 5. Final Accuracy (Test)

correct_test, total_test = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
final_test_acc = 100.0 * correct_test / total_test

print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Testing Accuracy:  {final_test_acc:.2f}%")
