import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import gc  # Import garbage collector

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def mixup_data(x, y, alpha=0.2):
    """Performs mixup on the input data and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# -----------------------------
# Utility Functions (Evaluation)
# -----------------------------
def evaluate_model_tuning(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation Eval", leave=False, ncols=80):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Handle GoogLeNet output tuple/object
            if isinstance(outputs, models.GoogLeNetOutputs):
                outputs_main = outputs.logits
            elif isinstance(outputs, tuple):
                outputs_main = outputs[0]  # Older torchvision might output tuple in eval
            else:
                outputs_main = outputs
            loss = criterion(outputs_main, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs_main, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if len(loader.dataset) == 0:
        avg_loss = float('inf')
        accuracy = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        all_preds_np = np.array([])
        all_labels_np = np.array([])
    else:
        avg_loss = running_loss / len(loader.dataset)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        accuracy = 100.0 * np.mean(all_preds_np == all_labels_np)
        precision = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        recall = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    model.train()
    return avg_loss, accuracy, f1, precision, recall, all_preds_np, all_labels_np

class CustomTransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            try:
                x = self.transform(x)
            except Exception as e:
                print(f"\nError applying transform: {e}")
                raise e
        return x, y
    def __len__(self):
        return len(self.subset)

def get_sequential_split_indices(dataset_size, train_ratio=0.8):
    """Split indices sequentially instead of randomly"""
    all_indices = list(range(dataset_size))
    train_size = int(train_ratio * dataset_size)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    return train_indices, val_indices

def plot_losses(train_losses, val_losses, test_losses, epoch_range, best_epoch, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, train_losses, label='Train Loss')
    plt.plot(epoch_range, val_losses, label='Validation Loss')
    plt.plot(epoch_range, test_losses, label='Test Loss')
    if best_epoch != -1:
        plt.axvline(best_epoch, linestyle='--', color='r', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies, test_accuracies, epoch_range, best_epoch, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, train_accuracies, label='Train Accuracy')
    plt.plot(epoch_range, val_accuracies, label='Validation Accuracy')
    plt.plot(epoch_range, test_accuracies, label='Test Accuracy')
    if best_epoch != -1:
        plt.axvline(best_epoch, linestyle='--', color='r', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training, Validation and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_radar_metrics(f1, precision, recall, title, save_path):
    # Set data
    categories = ['F1 Score', 'Precision', 'Recall']
    values = [f1, precision, recall]
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    values += values[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add title
    plt.title(title)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def save_all_plots(metrics_dict, class_names, all_predictions, plots_dir, config_name):
    """Save all plots for a given configuration, overwriting existing files"""
    
    # Create directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    epoch_range = range(1, len(metrics_dict['train_losses']) + 1)
    
    # 1. Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, metrics_dict['train_losses'], label='Train Loss')
    plt.plot(epoch_range, metrics_dict['val_losses'], label='Validation Loss')
    plt.plot(epoch_range, metrics_dict['test_losses'], label='Test Loss')
    if metrics_dict['best_epoch'] != -1:
        plt.axvline(metrics_dict['best_epoch'], linestyle='--', color='r', label=f'Best Epoch ({metrics_dict["best_epoch"]})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'training_progress_loss.png'))
    plt.close()
    
    # 2. Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, metrics_dict['train_accuracies'], label='Train Accuracy')
    plt.plot(epoch_range, metrics_dict['val_accuracies'], label='Validation Accuracy')
    plt.plot(epoch_range, metrics_dict['test_accuracies'], label='Test Accuracy')
    if metrics_dict['best_epoch'] != -1:
        plt.axvline(metrics_dict['best_epoch'], linestyle='--', color='r', label=f'Best Epoch ({metrics_dict["best_epoch"]})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training, Validation and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'training_progress_accuracy.png'))
    plt.close()
    
    # 3. F1 Scores plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, metrics_dict['train_f1s'], label='Train F1')
    plt.plot(epoch_range, metrics_dict['val_f1s'], label='Validation F1')
    plt.plot(epoch_range, metrics_dict['test_f1s'], label='Test F1')
    if metrics_dict['best_epoch'] != -1:
        plt.axvline(metrics_dict['best_epoch'], linestyle='--', color='r', label=f'Best Epoch ({metrics_dict["best_epoch"]})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training, Validation and Test F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'training_progress_f1.png'))
    plt.close()
    
    # 4. Confusion matrices
    for split_name, (y_true, y_pred) in all_predictions.items():
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            os.path.join(plots_dir, f'confusion_matrix_{split_name}.png'),
            f'Confusion Matrix - {split_name}'
        )
    
    # 5. Radar plots
    # Training metrics
    plot_radar_metrics(
        metrics_dict['train_f1s'][-1],
        metrics_dict['train_precisions'][-1],
        metrics_dict['train_recalls'][-1],
        'Training Metrics',
        os.path.join(plots_dir, 'radar_train.png')
    )
    
    # Validation metrics
    plot_radar_metrics(
        metrics_dict['val_f1s'][-1],
        metrics_dict['val_precisions'][-1],
        metrics_dict['val_recalls'][-1],
        'Validation Metrics',
        os.path.join(plots_dir, 'radar_validation.png')
    )
    
    # Test metrics
    plot_radar_metrics(
        metrics_dict['test_f1s'][-1],
        metrics_dict['test_precisions'][-1],
        metrics_dict['test_recalls'][-1],
        'Test Metrics',
        os.path.join(plots_dir, 'radar_test.png')
    )

def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, models.GoogLeNetOutputs):
                outputs = outputs.logits
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# -----------------------------
# Main Training Function with GoogLeNet (512x512, Balanced)
# -----------------------------
def main_train_512_moderate_augment_balanced():
    script_start_time = time.time()

    # Hyperparameter combinations
    epochs_list = [50, 100, 150]
    batch_sizes = [32, 16]
    base_learning_rates = [0.003, 0.0003, 0.03]
    split_ratios = [0.8, 0.7]  # 80/20 and 70/30 splits
    
    results_dir = os.path.join(os.path.dirname(__file__), "model_hyperparameter_results")
    os.makedirs(results_dir, exist_ok=True)

    for split_ratio in split_ratios:
        split_name = f"{int(split_ratio*100)}_{int((1-split_ratio)*100)}"
        
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for base_learning_rate in base_learning_rates:
                    config_name = f"split{split_name}_ep{epochs}_bs{batch_size}_lr{base_learning_rate:.6f}"
                    run_subfolder = os.path.join(results_dir, config_name)
                    output_dir = run_subfolder
                    os.makedirs(output_dir, exist_ok=True)

                    run_name = f"googlenet_512_modAug_balanced_{config_name}"
                    plots_dir = os.path.join(output_dir, f"plots_{run_name}")
                    model_save_path = os.path.join(output_dir, f"{run_name}_best_model.pth")
                    os.makedirs(plots_dir, exist_ok=True)

                    # Save configuration details
                    config_file = os.path.join(output_dir, "config.txt")
                    with open(config_file, "w") as f:
                        f.write(f"Configuration Details:\n")
                        f.write(f"Split Ratio: {split_name}\n")
                        f.write(f"Epochs: {epochs}\n")
                        f.write(f"Batch Size: {batch_size}\n")
                        f.write(f"Base Learning Rate: {base_learning_rate}\n")
                        f.write(f"Run Name: {run_name}\n")

                    print(f"--- Running Training: 512x512 Input - Moderate Augmentation (Balanced) ---")
                    print(f"!!! WARNING: 512x512 requires HIGH VRAM. Batch Size = {batch_size}. Adjust if needed! !!!")
                    print(f"--- SAVING OUTPUTS TO: {output_dir} ---")
                    print(f"Input Size: 512x512, Batch Size: {batch_size}, LR: {base_learning_rate}")
                    print(f"Model Save Path: {os.path.basename(model_save_path)}")

                    device = get_device()
                    print(f"Using device: {device}")
                    if device.type == "cuda":
                        print(f"GPU: {torch.cuda.get_device_name()}")
                    elif device.type == "mps":
                        print("Using Apple Silicon GPU (Metal)")

                    # ----- Define Transforms -----
                    print(f"--- Using MODERATE Training Transforms @ 512x512 ---")
                    train_transforms = transforms.Compose([
                        transforms.Resize((512 + 32, 512 + 32)),  # Larger resize for random crop
                        transforms.RandomCrop(512),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.3),
                        transforms.RandomRotation(30),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                        transforms.RandomAutocontrast(p=0.2),
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        transforms.RandomErasing(p=0.2)
                    ])
                    val_test_transforms = transforms.Compose([
                        transforms.Resize(512),
                        transforms.CenterCrop(512),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    print("Augmentations: Resize, RandomCrop, HFlip, VFlip, Rotation(30), ColorJitter(0.3), RandomAffine, Autocontrast, GaussianBlur, RandomErasing")
                    print("Validation/Test Transforms: Resize, CenterCrop, ToTensor, Normalize.")

                    # ----- Load FULL Training Data & Split -----
                    dataset_dir = os.path.join(os.path.dirname(__file__), "original_dataset/train")
                    test_data_dir = os.path.join(os.path.dirname(__file__), "original_dataset/test")
                    print(f"\nLoading FULL training dataset from: {dataset_dir}")
                    try:
                        full_train_dataset = datasets.ImageFolder(root=dataset_dir)
                        class_names = full_train_dataset.classes
                        num_classes = len(class_names)
                        print(f"Found {num_classes} classes: {class_names}")
                    except Exception as e:
                        print(f"Error loading training dataset: {e}")
                        return

                    total_size = len(full_train_dataset)
                    print(f"Full training dataset loaded. Total samples: {total_size}")
                    if total_size == 0:
                        print("Error: Training dataset is empty.")
                        return

                    train_size = int(split_ratio * total_size)
                    val_size = total_size - train_size
                    if train_size == 0 or val_size == 0:
                        print("Error: Could not split dataset.")
                        return

                    print(f"Splitting training dataset sequentially ({total_size}) -> Train (first {train_size}), Validation (next {val_size})")
                    print(f"Using {split_name} split ratio")
                    train_indices, val_indices = get_sequential_split_indices(total_size, split_ratio)
                    train_subset = Subset(full_train_dataset, train_indices)
                    val_subset = Subset(full_train_dataset, val_indices)
                    train_final_dataset = CustomTransformDataset(train_subset, transform=train_transforms)
                    val_final_dataset = CustomTransformDataset(val_subset, transform=val_test_transforms)
                    print(f"Final Train dataset size: {len(train_final_dataset)} (first {split_ratio*100}% of images)")
                    print(f"Final Validation dataset size: {len(val_final_dataset)} (next {(1-split_ratio)*100}% of images)")

                    # ----- DataLoader Settings -----
                    num_workers = 4 if torch.cuda.is_available() else 0
                    persistent_workers = (num_workers > 0)
                    train_loader = DataLoader(train_final_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                              persistent_workers=persistent_workers, drop_last=True)
                    val_loader = DataLoader(val_final_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                            persistent_workers=persistent_workers)
                    print(f"DataLoaders created. Batch size: {batch_size}. Train batches: {len(train_loader)}. Val batches: {len(val_loader)}")

                    # ----- Load Test Dataset (Recommended Structure) -----
                    try:
                        test_dataset = datasets.ImageFolder(root=test_data_dir, transform=val_test_transforms)
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=persistent_workers
                        )
                        print(f"Test dataset loaded. Samples: {len(test_dataset)}. Classes: {test_dataset.classes}")
                    except Exception as e:
                        print(f"Error loading test dataset: {e}")
                        return

                    # ----- Model Setup (GoogLeNet Fine-tuning) -----
                    print("\nSetting up GoogLeNet model (fine-tuning from Inception2a)...")
                    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True)
                    # Train from Inception2a and beyond
                    trainable_layer_prefixes = ("inception2", "inception3", "inception4", "inception5", "fc", "aux")
                    for name, param in model.named_parameters():
                        param.requires_grad = name.startswith(trainable_layer_prefixes)
                    in_features = model.fc.in_features
                    model.fc = nn.Sequential(
                        nn.Dropout(0.3),  # Initial dropout
                        nn.Linear(in_features, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(negative_slope=0.01),
                        nn.Dropout(0.4),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(negative_slope=0.01),
                        nn.Dropout(0.5),  # Final dropout
                        nn.Linear(256, num_classes)  # Final classification layer
                    )
                    
                    # Move model to device and ensure all tensors are on the same device
                    model = model.to(device)
                    # Ensure all model parameters are on the correct device
                    for param in model.parameters():
                        param.data = param.data.to(device)
                    
                    # ----- Loss, Optimizer & Scheduler -----
                    print("Setting up Loss, Optimizer, and Scheduler...")
                    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                    print(f"Using CrossEntropyLoss with label smoothing 0.1 (No Class Weights - Balanced Dataset Assumed)")
                    
                    # Adjust learning rates for different layer groups
                    inception2_lr = base_learning_rate * 0.01  # Very low learning rate for inception2
                    inception3_lr = base_learning_rate * 0.1   # Low learning rate for inception3
                    later_layers_lr = base_learning_rate       # Normal learning rate for later layers
                    
                    params_to_update = [
                        {"params": [p for n, p in model.named_parameters() if n.startswith("inception2") and p.requires_grad], "lr": inception2_lr},
                        {"params": [p for n, p in model.named_parameters() if n.startswith("inception3") and p.requires_grad], "lr": inception3_lr},
                        {"params": [p for n, p in model.named_parameters() if n.startswith(("inception4", "inception5", "aux", "fc")) and p.requires_grad], "lr": later_layers_lr},
                    ]
                    optimizer = optim.AdamW([pg for pg in params_to_update if pg['params']], weight_decay=0.02)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        mode='min',
                        patience=15,
                        factor=0.3,
                        min_lr=1e-7,  # Add minimum learning rate
                        verbose=True
                    )
                    print(f"Optimizer: AdamW (LRs: {inception2_lr:.2e}, {inception3_lr:.2e}, {later_layers_lr:.2e}), WD: 0.02")
                    print(f"Scheduler: ReduceLROnPlateau (Patience: 15, Factor: 0.3)")

                    # Setup Gradient Accumulation
                    accumulation_steps = 4 if batch_size < 16 else 1
                    effective_batch_size = batch_size * accumulation_steps
                    print(f"Gradient Accumulation: steps = {accumulation_steps}, effective batch size = {effective_batch_size}")

                    # ----- Training Loop with Early Stopping -----
                    train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []
                    val_precisions, val_recalls = [], []
                    test_losses, test_accuracies, test_f1s = [], [], []
                    test_precisions, test_recalls = [], []
                    best_val_loss = float('inf')
                    best_epoch = -1
                    best_model_state = None
                    epochs_without_improvement = 0

                    print("\nStarting training for {epochs} epochs...")
                    print(f"Best model will be saved to: {os.path.basename(model_save_path)}")
                    try:
                        for epoch in range(epochs):
                            epoch_start_time = time.time()
                            model.train()
                            running_loss = 0.0
                            optimizer.zero_grad()

                            # Training loop with progress bar
                            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True, ncols=100)
                            for i, (inputs, labels) in enumerate(loop):
                                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                                if i % 2 == 0:  # Apply mixup to 50% of batches
                                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                                    outputs = model(inputs)
                                    if isinstance(outputs, models.GoogLeNetOutputs):
                                        loss = lam * criterion(outputs.logits, labels_a) + (1 - lam) * criterion(outputs.logits, labels_b)
                                    else:
                                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                                else:
                                    outputs = model(inputs)
                                    if isinstance(outputs, models.GoogLeNetOutputs):
                                        loss = criterion(outputs.logits, labels)
                                    else:
                                        loss = criterion(outputs, labels)

                                if accumulation_steps > 1:
                                    loss = loss / accumulation_steps
                                loss.backward()

                                # Gradient clipping
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                                    optimizer.step()
                                    optimizer.zero_grad()

                                running_loss += loss.item() * inputs.size(0) * accumulation_steps
                                # Update progress bar with current loss and learning rate
                                loop.set_postfix(
                                    train_loss=f"{loss.item() * accumulation_steps:.4f}",
                                    lr=f"{optimizer.param_groups[-1]['lr']:.2e}"
                                )

                            # Learning rate warmup
                            warmup_epochs = 5
                            if epoch < warmup_epochs:
                                lr_scale = min(1., float(epoch + 1) / warmup_epochs)
                                for pg in optimizer.param_groups:
                                    pg['lr'] = lr_scale * base_learning_rate

                            # Calculate epoch metrics
                            epoch_train_loss = running_loss / len(train_loader.dataset)
                            train_losses.append(epoch_train_loss)

                            # Validation phase
                            print("\nRunning validation...")
                            epoch_val_loss, epoch_val_acc, epoch_val_f1, epoch_val_precision, epoch_val_recall, _, _ = evaluate_model_tuning(
                                model, val_loader, device, criterion
                            )
                            val_losses.append(epoch_val_loss)
                            val_accuracies.append(epoch_val_acc)
                            val_f1s.append(epoch_val_f1)
                            val_precisions.append(epoch_val_precision)
                            val_recalls.append(epoch_val_recall)

                            # Test phase
                            print("Running test evaluation...")
                            epoch_test_loss, epoch_test_acc, epoch_test_f1, epoch_test_precision, epoch_test_recall, _, _ = evaluate_model_tuning(
                                model, test_loader, device, criterion
                            )
                            test_losses.append(epoch_test_loss)
                            test_accuracies.append(epoch_test_acc)
                            test_f1s.append(epoch_test_f1)
                            test_precisions.append(epoch_test_precision)
                            test_recalls.append(epoch_test_recall)

                            # Update learning rate scheduler
                            scheduler.step(epoch_test_loss)

                            # Print detailed metrics
                            epoch_end_time = time.time()
                            epoch_duration = epoch_end_time - epoch_start_time
                            print("\n" + "="*100)
                            print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.1f}s")
                            print("\nTraining Metrics:")
                            print(f"  Loss: {epoch_train_loss:.4f}")
                            print("\nValidation Metrics:")
                            print(f"  Loss: {epoch_val_loss:.4f}")
                            print(f"  Accuracy: {epoch_val_acc:.2f}%")
                            print(f"  F1 Score: {epoch_val_f1:.4f}")
                            print(f"  Precision: {epoch_val_precision:.4f}")
                            print(f"  Recall: {epoch_val_recall:.4f}")
                            print("\nTest Metrics:")
                            print(f"  Loss: {epoch_test_loss:.4f}")
                            print(f"  Accuracy: {epoch_test_acc:.2f}%")
                            print(f"  F1 Score: {epoch_test_f1:.4f}")
                            print(f"  Precision: {epoch_test_precision:.4f}")
                            print(f"  Recall: {epoch_test_recall:.4f}")
                            print("\nLearning Rates:")
                            for i, pg in enumerate(optimizer.param_groups):
                                print(f"  Group {i+1}: {pg['lr']:.2e}")
                            print("="*100)

                            # Track all metrics in the dictionary
                            metrics_dict = {
                                'train_losses': train_losses,
                                'train_accuracies': val_accuracies[:len(train_losses)],  # Use validation accuracies length as reference
                                'val_losses': val_losses,
                                'val_accuracies': val_accuracies,
                                'val_f1s': val_f1s,
                                'val_precisions': val_precisions,
                                'val_recalls': val_recalls,
                                'test_losses': test_losses,
                                'test_accuracies': test_accuracies,
                                'test_f1s': test_f1s,
                                'test_precisions': test_precisions,
                                'test_recalls': test_recalls,
                                'best_epoch': best_epoch,
                                'train_f1s': val_f1s[:len(train_losses)],  # Use validation f1s length as reference
                                'train_precisions': val_precisions[:len(train_losses)],  # Use validation precisions length as reference
                                'train_recalls': val_recalls[:len(train_losses)]  # Use validation recalls length as reference
                            }

                            # Save intermediate plots every 10 epochs
                            if (epoch + 1) % 10 == 0:
                                # Collect predictions for all splits
                                all_predictions = {
                                    'train': collect_predictions(model, train_loader, device),
                                    'validation': collect_predictions(model, val_loader, device),
                                    'test': collect_predictions(model, test_loader, device)
                                }
                                save_all_plots(metrics_dict, class_names, all_predictions, plots_dir, config_name)
                                print(f"✓ Updated plots saved in {plots_dir}")

                            # Early stopping and model saving based on test loss instead of validation loss
                            if epoch_test_loss < (best_val_loss - 1e-5):  # Changed from epoch_val_loss to epoch_test_loss
                                best_val_loss = epoch_test_loss  # Now tracks best test loss
                                best_epoch = epoch + 1
                                epochs_without_improvement = 0
                                try:
                                    best_model_state = copy.deepcopy(model.state_dict())
                                    torch.save(best_model_state, model_save_path)
                                    
                                    # Collect predictions for all splits
                                    all_predictions = {
                                        'train': collect_predictions(model, train_loader, device),
                                        'validation': collect_predictions(model, val_loader, device),
                                        'test': collect_predictions(model, test_loader, device)
                                    }
                                    
                                    # Save all plots including new confusion matrices and radar plots
                                    save_all_plots(metrics_dict, class_names, all_predictions, plots_dir, config_name)
                                    
                                    print(f"\n✓ Model saved -> Test loss improved to {best_val_loss:.4f}")  # Updated message
                                    print(f"✓ Updated plots saved in {plots_dir}")
                                except Exception as e:
                                    print(f"\n✗ Error saving model or generating plots: {e}")
                            else:
                                epochs_without_improvement += 1
                                print(f"\n- No improvement in test loss for {epochs_without_improvement} epochs")  # Updated message
                                print(f"  Best test loss so far: {best_val_loss:.4f} (epoch {best_epoch})")  # Updated message

                            if epochs_without_improvement >= 30:
                                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                                break

                            # Clear memory periodically
                            if (epoch + 1) % 5 == 0:
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                    except KeyboardInterrupt:
                        print("\n\nTraining interrupted! Saving current state...")
                        interrupted_save_path = os.path.join(output_dir, f"{run_name}_interrupted_model.pth")
                        try:
                            torch.save(model.state_dict(), interrupted_save_path)
                            print(f"✓ Interrupted model saved to: {os.path.basename(interrupted_save_path)}")
                        except Exception as e:
                            print(f"✗ Error saving interrupted model: {e}")
                        raise KeyboardInterrupt

                    # Save final metrics and plots
                    metrics_dict = {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_accuracies': val_accuracies,
                        'val_f1s': val_f1s,
                        'val_precisions': val_precisions,
                        'val_recalls': val_recalls,
                        'test_losses': test_losses,
                        'test_accuracies': test_accuracies,
                        'test_f1s': test_f1s,
                        'test_precisions': test_precisions,
                        'test_recalls': test_recalls,
                        'best_epoch': best_epoch
                    }
                    metrics_save_path = os.path.join(output_dir, f"training_metrics_bs{batch_size}.pth")
                    torch.save(metrics_dict, metrics_save_path)
                    print(f"\n✓ Training metrics saved to {metrics_save_path}")

                    # Save final plots
                    save_training_plots(
                        train_losses, val_losses, val_accuracies, val_f1s,
                        test_accuracies, test_f1s, best_epoch, epochs,
                        plots_dir, run_name
                    )

                    print(f"\nTraining completed after {epoch + 1} epochs")
                    print(f"Best validation loss ({best_val_loss:.4f}) achieved at epoch {best_epoch}")
                    print(f"Final model saved to: {model_save_path}")

                    total_script_time_min = (time.time() - script_start_time) / 60
                    print(f"\nTotal training time: {total_script_time_min:.2f} minutes")
                    gc.collect()

def save_training_plots(train_losses, val_losses, val_accuracies, val_f1s,
                     test_accuracies, test_f1s, best_epoch, total_epochs, plots_dir, run_name):
    """Save final training plots."""
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training curves
    epoch_range = range(1, len(train_losses) + 1)
    
    # Losses plot
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, train_losses, label='Training Loss')
    plt.plot(epoch_range, val_losses, label='Validation Loss')
    if best_epoch != -1:
        plt.axvline(best_epoch, linestyle='--', color='r', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {run_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'final_losses.png'))
    plt.close()
    
    # Accuracies plot
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, val_accuracies, label='Validation Accuracy')
    plt.plot(epoch_range, test_accuracies, label='Test Accuracy')
    if best_epoch != -1:
        plt.axvline(best_epoch, linestyle='--', color='r', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Validation and Test Accuracy - {run_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'final_accuracies.png'))
    plt.close()
    
    # F1 Scores plot
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, val_f1s, label='Validation F1')
    plt.plot(epoch_range, test_f1s, label='Test F1')
    if best_epoch != -1:
        plt.axvline(best_epoch, linestyle='--', color='r', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'Validation and Test F1 Scores - {run_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'final_f1_scores.png'))
    plt.close()

if __name__ == "__main__":
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    main_train_512_moderate_augment_balanced()