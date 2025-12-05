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
import torch.optim as optim
from tqdm import tqdm

dataset = pd.read_csv('data/train.csv')

# create custom dataset
class APTOSDataset(Dataset):
    def __init__(self, csv_df, transform=None):
        self.df = csv_df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(os.path.join('data/train_images', f'{self.df.iloc[index]["id_code"]}.png'))
        label = self.df.iloc[index]['diagnosis']

        if self.transform:
            image = self.transform(image)

        return image, label

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

# create train and test dataloaders
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

# define the model
NUM_CLASSES = 5

model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# freeze all the parameters
for name, param in model_resnet.named_parameters():
    param.requires_grad = False

# replace the last fc classifier layer
fc_num_in_features = model_resnet.fc.in_features

model_resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(fc_num_in_features, NUM_CLASSES)
)

# unfreeze last layer's parameters
for name, param in model_resnet.fc.named_parameters():
    param.requires_grad = True

# extract the parameters that will be trained
params_to_update = [param for param in model_resnet.parameters() if param.requires_grad]

# define the optimizer and the loss function
LEARNING_RATE = 1e-3  # A standard LR for training a small head
optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_resnet.to(device)

# helper function to evaluate the model
def evaluate_model(model, data_loader, criterion, device):
    
    model.eval() 
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # run without tracking gradients
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            
            running_loss += loss.item() * inputs.size(0)
            
            # find the predicted class (index of the max logit)
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # calculate final metrics
    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    # set the model back to training mode after evaluation is done
    model.train() 
    
    return avg_loss, accuracy


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):

    # track loss and accuracy over time
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print("--- Beginning training...")

    for epoch in range(num_epochs):
        
        model.train() 

        running_loss = 0.0 
        correct_predictions = 0
        total_samples = 0

        TRAIN_DATASET_SIZE = len(train_dataset)
        
        for inputs, labels in tqdm(train_loader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero out gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, labels.long())
            
            # backpropagation
            loss.backward()
            
            # update parameters
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # print stats per epoch
        epoch_train_loss = running_loss / TRAIN_DATASET_SIZE
        epoch_train_acc = correct_predictions / total_samples
        epoch_test_loss, epoch_test_acc = evaluate_model(model, test_loader, criterion, device)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} - Test Loss: {epoch_test_loss:.4f} - Test Acc: {epoch_test_acc:.4f}')

    print("--- Training complete")
    return history

history = train_model(model_resnet, train_loader, criterion, optimizer, device, 1)
print(history)

MODEL_DIR = 'tmodel_checkpoints'
MODEL_FILENAME = 'resnet50_stage1_baseline.pth'
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(model_resnet.state_dict(), SAVE_PATH)

# reload model for stage 2
loaded_state_dict = torch.load(SAVE_PATH, map_location=device)

model_resnet.load_state_dict(loaded_state_dict)

model_resnet.to(device)

for param in model_resnet.layer4.parameters():
    param.requires_grad = True

params_to_update_stage2 = [param for param in model_resnet.parameters() if param.requires_grad]

FINETUNING_LEARNING_RATE = 1e-5
optimizer_stage2 = optim.Adam(params_to_update_stage2, lr=FINETUNING_LEARNING_RATE)

criterion = nn.CrossEntropyLoss()

FINETUNING_EPOCHS = 1

metrics_history_stage2 = train_model(
    model=model_resnet, 
    train_loader=train_loader, 
    criterion=criterion, 
    optimizer=optimizer_stage2,  # Use the new optimizer
    device=device, 
    num_epochs=FINETUNING_EPOCHS
)

MODEL_FILENAME = 'resnet50_stage2_baseline.pth'
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model_resnet.state_dict(), SAVE_PATH)
