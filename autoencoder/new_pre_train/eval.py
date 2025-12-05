import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from MAEVitModel import MAEViTModel
from FineTunedViT import FineTunedViT
from main import CustomImageDataset, train_test_transform  
from sklearn.metrics import precision_recall_fscore_support

BATCH_SIZE = 32
NUM_CLASSES = 5
FILENAME_COLUMN = "id_code"
LABEL_COLUMN    = "diagnosis"
IMAGE_SUFFIX    = ".png"
TRAIN_CSV_PATH = "/home/jobe/ECS271/project/dataset/aptos/train_split.csv" 
TEST_CSV_PATH  = "/home/jobe/ECS271/project/dataset/aptos/test_split.csv"
DATA_ROOT      = "/home/jobe/ECS271/project/dataset/aptos/train_images"
# WEIGHTS_PATH   = "/home/jobe/ECS271/project/weights/mae_vit.safetensors" 
WEIGHTS_PATH   = "/home/jobe/ECS271/ECS-271-F25-Project/weights/vit_epoch10.pth" 
JSON_RESULTS = "/home/jobe/ECS271/ECS-271-F25-Project/evaluation_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_dataset, batch_size, criterion, device, num_classes=NUM_CLASSES):
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    running_loss = 0.0
    total_samples = 0
    class_stats = torch.zeros(num_classes, 2)  # correct, total

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

            correct = (predicted == labels)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_stats[label, 1] += 1
                if correct[i].item():
                    class_stats[label, 0] += 1
            total_samples += labels.size(0)

    final_preds = torch.cat(all_preds).numpy()
    final_labels = torch.cat(all_labels).numpy()

    avg_loss = running_loss / total_samples
    overall_correct = torch.sum(class_stats[:,0]).item()
    overall_accuracy = overall_correct / total_samples

    class_accuracy = {}
    for i in range(num_classes):
        total_count = class_stats[i,1].item()
        class_accuracy[i] = class_stats[i,0].item() / total_count if total_count > 0 else 0.0

    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        final_labels, final_preds,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0
    )

    per_class_metrics = {
        'precision': {i: precision_arr[i] for i in range(num_classes)},
        'recall': {i: recall_arr[i] for i in range(num_classes)},
        'f1': {i: f1_arr[i] for i in range(num_classes)},
    }

    return avg_loss, overall_accuracy, class_accuracy, per_class_metrics

def main():
    test_dataset = CustomImageDataset(
        csv_file=TEST_CSV_PATH,
        img_dir=DATA_ROOT,
        transform=train_test_transform,
        filename_col=FILENAME_COLUMN,
        label_col=LABEL_COLUMN,
        suffix=IMAGE_SUFFIX,
    )

    # model = MAEViTModel(num_classes=NUM_CLASSES)
    model = FineTunedViT(num_classes=NUM_CLASSES)
    model = model.load_trained_weights(WEIGHTS_PATH, device=DEVICE)
    model.to(DEVICE)
    # model_name = "MAE ViT"
    model_name = "Finetuned ViT"

    criterion = nn.CrossEntropyLoss()

    avg_loss, overall_accuracy, class_accuracy, per_class_metrics = evaluate_model(
        model, test_dataset, BATCH_SIZE, criterion, DEVICE
    )

    print(f"\nAverage Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
    print(f"Per-Class Accuracy: {class_accuracy}")
    print(f"Per-Class Metrics (Precision / Recall / F1): {per_class_metrics}")

    # Load existing results or start new
    if os.path.exists(JSON_RESULTS):
        try:
            with open(JSON_RESULTS, "r") as f:
                results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # File exists but is empty or corrupted, start fresh
            results = {}

    # Update or add entry for this model
    results[model_name] = {
        "avg_loss": avg_loss,
        "overall_accuracy": overall_accuracy,
        "class_accuracy": class_accuracy,
        "per_class_metrics": per_class_metrics
    }

    # Save back to JSON
    with open(JSON_RESULTS, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()