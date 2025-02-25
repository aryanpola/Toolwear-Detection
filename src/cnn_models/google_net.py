import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    # 1. Configuration

    dataset_dir = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\original_dataset\train"
    results_file = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\model_evaluation.txt"  # File to save results
    num_classes = 3
    batch_size = 16
    learning_rate = 0.0001
    num_epochs = 50
    use_scheduler = True

    # 2. Data Transformations

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Load Dataset & Split

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transforms)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 4. Device Configuration

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 5. Load & Modify Pre-Trained GoogLeNet

    model = models.googlenet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    if hasattr(model, 'aux1') and model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    if hasattr(model, 'aux2') and model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

    model = model.to(device)

    # 6. Loss, Optimizer, and Scheduler

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 7. Training Loop

    def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
        model.train()
        training_loss_history = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    main_outputs, aux1, aux2 = outputs
                    loss1 = criterion(main_outputs, labels)
                    loss2 = criterion(aux1, labels)
                    loss3 = criterion(aux2, labels)
                    loss = loss1 + 0.3 * (loss2 + loss3)
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)
            training_loss_history.append(avg_loss)
            print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")

            if use_scheduler:
                scheduler.step()

        print("Training complete.")
        return training_loss_history

    # 8. Evaluation Function (Save to File)

    def evaluate_model(model, test_loader, device, results_file):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save to file
        with open(results_file, "w") as f:
            f.write("Model Evaluation Results\n")
            f.write("=======================\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        print(f"\nResults saved to: {results_file}")

        return accuracy, precision, recall, f1


    # 9. Run Training & Evaluation

    training_loss_history = train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)

    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device, results_file)

    # 10. Save the Model

    save_path = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\GoogLeNet_augmented.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
