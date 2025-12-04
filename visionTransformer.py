import torch
from vit_pytorch import ViT
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision.models as models
from collections import OrderedDict

TRAIN_DATA_ROOT = './data/aptos2019-blindness-detection/train_images'
TEST_DATA_ROOT = './data/aptos2019-blindness-detection/test_images'
DATA_ROOT = './data/aptos2019-blindness-detection/images'
TRAIN_CSV_PATH = './data/aptos2019-blindness-detection/train.csv'
TEST_CSV_PATH = './data/aptos2019-blindness-detection/test.csv'
WEIGHTS_PATH = './weights/vit_epoch10.pth'
FILENAME_COLUMN = 'id_code'
LABEL_COLUMN = 'diagnosis'
IMAGE_SUFFIX = '.png'
"""
Functional Dataset class to load images and labels from a CSV file and image directory.

Params:
    csv_file (str): Path to the CSV file with annotations.
    img_dir (str): Directory with all the images.
    transform (callable, optional): Optional transform to be applied on a sample.
    filename_col (str): Name of the column containing image IDs/filenames.
    label_col (str): Name of the column containing class labels.
    suffix (str): File extension to append to the ID (e.g., '.jpg').
"""
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, filename_col='id_code', label_col='diagnosis', suffix='.png'):
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.filename_col = filename_col
        self.label_col = label_col
        self.suffix = suffix
        
        # Make sure labels are integers
        self.annotations[self.label_col] = self.annotations[self.label_col].astype(int)

        # Print label distribution for safety
        print("Classes found:", sorted(self.annotations[self.label_col].unique()))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 1. Get the image ID and label from the CSV row
        row = self.annotations.iloc[idx]
        img_id = str(row[self.filename_col])
        label = int(row[self.label_col])
        
        # 2. Construct the full path and load the image
        img_path = os.path.join(self.img_dir, img_id + self.suffix)
        
        # Ensure the image file exists (critical for debugging file issues)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        image = Image.open(img_path).convert('RGB') # Load and ensure 3 color channels

        # 3. Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    NUM_CLASSES = 5
    

    train_and_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    # Load custom datasets using the new CustomImageDataset class
    print(f"\nAttempting to load data using custom CSV/Folder structure...")
    try:
        train_dataset = CustomImageDataset(
            csv_file=TRAIN_CSV_PATH, 
            img_dir=DATA_ROOT, 
            transform=train_and_test_transform,
            filename_col=FILENAME_COLUMN,
            label_col=LABEL_COLUMN,
            suffix=IMAGE_SUFFIX
        )
        test_dataset = CustomImageDataset(
            csv_file=TEST_CSV_PATH, 
            img_dir=DATA_ROOT, 
            transform=train_and_test_transform,
            filename_col=FILENAME_COLUMN,
            label_col=LABEL_COLUMN,
            suffix=IMAGE_SUFFIX
        )
    except FileNotFoundError as e:
        # Custom path not found
        print("WARNING: Custom data paths or CSV files not found.")
        print(f"Error: {e}")
        print("Please replace DATA_ROOT with your actual path.")

    
    print(f"\nModel will be trained on {NUM_CLASSES} classes.")

    # Create DataLoaders for batching
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    NUM_CLASSES = 5
    classes = (0, 1, 2, 3, 4)

    net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    print("Loaded ViT-Base/16 with ImageNet pre-trained weights.")
    
    NUM_CLASSES = 5
    net.heads.head = torch.nn.Linear(net.heads.head.in_features, NUM_CLASSES)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    LR_STAGE1 = 1e-3  # Use a slightly higher LR for the new head
    EPOCHS_STAGE1 = 3 # Train for a few epochs

    # Freeze all parameters
    for param in net.parameters():
        param.requires_grad = False

    # Unfreeze ONLY the classification head
    for param in net.heads.head.parameters():
        param.requires_grad = True

    # Re-define the optimizer to include only the trainable parameters (the head)
    optimizer_stage1 = optim.SGD(net.heads.head.parameters(), lr=LR_STAGE1, momentum=0.9)

    print("Fine-tuning last layer with higher learning rate.")
    for epoch in range(1):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    print("Training whole model now.")
    LR_STAGE2 = 1e-5 # Use a very small LR for fine-tuning the whole model
    EPOCHS_STAGE2 = 15 # Train for more epochs

    # Unfreeze all parameters
    for param in net.parameters():
        param.requires_grad = True

    # Define the final optimizer with tiny LR for the entire network
    optimizer = optim.SGD(net.parameters(), lr=LR_STAGE2, momentum=0.9)

    #Weights initialized to handle class imbalance
    loss_weights_tensor = torch.tensor([1.0, 4.9, 1.8, 9.4, 6.1], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=loss_weights_tensor.to(device)) 

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), WEIGHTS_PATH)
    print("Model weights saved successfully to:", WEIGHTS_PATH)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 732 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * float(class_correct[i]) / float(class_total[i])))