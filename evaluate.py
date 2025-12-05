import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# MODIFY IF NECESSARY
IMAGES_PATH = "data/train_images"
CSV_PATH = "data/train.csv"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = pd.read_csv(CSV_PATH)

    # create custom dataset
    class APTOSDataset(Dataset):
        def __init__(self, csv_df, transform=None):
            self.df = csv_df
            self.transform = transform

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, index):
            image = Image.open(os.path.join(IMAGES_PATH, f'{self.df.iloc[index]["id_code"]}.png'))
            label = self.df.iloc[index]['diagnosis']

            if self.transform:
                image = self.transform(image)

            return image, label

    IMAGE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    NUM_CLASSES = 5

    training_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # randomly split the dataset into 80% train / 20% test
    train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset['diagnosis'], random_state=42)

    # create train and test datasets
    train_dataset = APTOSDataset(train_df, training_transform)
    test_dataset = APTOSDataset(test_df, test_transform)

    models = {}

    # add the Resnet model
    model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    fc_num_in_features = model_resnet.fc.in_features # replace the last fc classifier layer
    model_resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(fc_num_in_features, NUM_CLASSES)
    )
    # load in trained weights
    WEIGHT_PATH = 'resnet50_stage2_baseline.pth'
    loaded_state_dict = torch.load(WEIGHT_PATH, map_location=device)
    model_resnet.load_state_dict(loaded_state_dict)
    model_resnet.to(device)
    models["Resnet50"] = [model_resnet, 32] # model, batch size - MAKE SURE TO ADD YOUR OWN BATCH SIZE
    
    # ADD YOUR MODELS HERE - instantiate, make any changes, load weights, send to device, add to models dict

    def evaluate_model(model, batch_size, criterion, device, num_classes=5):

        data_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

        #  set the model to evaluation mode
        # model.eval() 
        
        running_loss = 0.0
        total_samples = 0
        
        class_stats = torch.zeros(num_classes, 2) # [correct, total]
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
                
                correct = (predicted == labels).squeeze()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_stats[label, 1] += 1
                    if correct[i].item():
                        class_stats[label, 0] += 1
                
                total_samples += labels.size(0)

        # combine all predictions and labels from al batches
        final_preds = torch.cat(all_preds).numpy()
        final_labels = torch.cat(all_labels).numpy()
        
        # overall loss and accuracy
        avg_loss = running_loss / total_samples
        overall_correct = torch.sum(class_stats[:, 0]).item()
        overall_accuracy = overall_correct / total_samples
        
        # per class accuracy
        class_accuracy = {}
        for i in range(num_classes):
            total_count = class_stats[i, 1].item()
            accuracy = class_stats[i, 0].item() / total_count if total_count > 0 else 0.0
            class_accuracy[i] = accuracy
            
        # per class precision, recall, f1
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


    # evaluate each model and store results - MAKE SURE TO USE YOUR BATCH SIZE
    results = {}
    for model_name, [model, batch_size] in models.items():
        avg_loss, overall_accuracy, class_accuracy, per_class_metrics = evaluate_model(model, batch_size, nn.CrossEntropyLoss(), device)
        results[model_name] = [avg_loss, overall_accuracy, class_accuracy, per_class_metrics]
        results["Resnet51"] = [avg_loss, overall_accuracy, class_accuracy, per_class_metrics]
        results["Resnet52"] = [avg_loss, overall_accuracy, class_accuracy, per_class_metrics]



    # graph results
    os.makedirs('figures', exist_ok=True)
    COLORS = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

    # plot average accuracy
    model_names = list(results.keys())
    overall_accuracies = [data[1] for data in results.values()]
    plt.clf()
    plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, overall_accuracies, color=COLORS[:len(model_names)])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(overall_accuracies) * 0.01), f'{yval:.4f}', ha='center', va='bottom')
    plt.ylabel('accuracy')
    plt.title("Average Accuracy")
    plt.savefig('figures/avg_accuracy.png')

    # plot average loss
    model_names = list(results.keys())
    average_losses = [data[0] for data in results.values()]
    plt.clf() 
    plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, average_losses, color=COLORS[:len(model_names)])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(average_losses) * 0.01), f'{yval:.4f}', ha='center', va='bottom')
    plt.ylabel('loss')
    plt.title("Average Loss")
    plt.tight_layout()
    plt.savefig('figures/avg_loss.png')

    # plot accuracy per class
    model_names = list(results.keys())
    num_models = len(model_names)
    num_classes = 5
    class_data = {}
    for i in range(num_classes):
        class_data[i] = [results[model][2][i] for model in model_names]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    bar_width = 0.25
    index = np.arange(1)
    for i in range(num_classes):
        ax = axes[i]
        accuracies = class_data[i]
        
        for j in range(num_models):
            ax.bar(index + j * bar_width, accuracies[j], bar_width, color=COLORS[j], label=model_names[j])
            
            yval = accuracies[j]
            ax.text(index[0] + j * bar_width, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Class {i} Accuracy', fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('accuracy', fontsize=12)
        ax.set_xticks(index + bar_width * (num_models - 1) / 2)
        ax.set_xticklabels([''])
        
    if num_classes < len(axes):
        fig.delaxes(axes[num_classes])

    fig.suptitle('Per-Class Accuracy For Each Model (Grades 0 to 4)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('figures/class_accuracy_grid.png')

    # plot precision, f1, recall per class
    model_names = list(results.keys())
    num_models = len(model_names)
    num_classes = 5
    metrics = ['precision', 'recall', 'f1']
    num_metrics = len(metrics)
    metric_data = {
        metric: {
            i: [results[model][3][metric][i] for model in model_names] 
            for i in range(num_classes)
        } 
        for metric in metrics
    }
    fig, axes = plt.subplots(num_metrics, num_classes, figsize=(20, 12), sharey=True)
    bar_width = 0.25
    index = np.arange(1)

    for row_idx, metric_name in enumerate(metrics):
        for col_idx in range(num_classes):
            ax = axes[row_idx, col_idx]
            scores = metric_data[metric_name][col_idx]
            
            for j in range(num_models):
                ax.bar(index + j * bar_width, scores[j], bar_width, color=COLORS[j])
                
                yval = scores[j]
                ax.text(index[0] + j * bar_width, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_ylim(0, 1.05)
            ax.set_xticks(index + bar_width * (num_models - 1) / 2)
            ax.set_xticklabels([''])
            
            if col_idx == 0:
                ax.set_ylabel(metric_name.capitalize(), fontsize=14, weight='bold')
                
            if row_idx == 0:
                ax.set_title(f'Class {col_idx}', fontsize=14)

    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=COLORS[j]) for j in range(num_models)],
        model_names,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=num_models,
        title='Model Architecture'
    )

    fig.suptitle('Per-Class Performance Comparison (Precision, Recall, F1)', fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('figures/class_metrics_grid.png')


if __name__ == "__main__":
    main()