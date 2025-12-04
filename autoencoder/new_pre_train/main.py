import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16

from safetensors.torch import load_file

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

# Paths
TRAIN_CSV_PATH = "/home/jobe/ECS271/project/dataset/aptos/train_split.csv" 
TEST_CSV_PATH  = "/home/jobe/ECS271/project/dataset/aptos/test_split.csv"
DATA_ROOT      = "/home/jobe/ECS271/project/dataset/aptos/train_images"
WEIGHTS_PATH   = "/home/jobe/ECS271/project/weights" 
FILENAME_COLUMN = "id_code"
LABEL_COLUMN    = "diagnosis"
IMAGE_SUFFIX    = ".png"

# Hyperparameters
BATCH_SIZE = 4
NUM_CLASSES = 5
EPOCHS_STAGE1 = 3
EPOCHS_STAGE2 = 15
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5

train_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

def train_head_only(net, trainloader, criterion, device):
    print("\nStage 1: Training classification head only...")
    for param in net.parameters():
        param.requires_grad = False
    for param in net.heads.head.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.SGD(net.heads.head.parameters(), lr=LR_STAGE1, momentum=0.9)

    for epoch in range(EPOCHS_STAGE1):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_stage1.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage1.step()

        print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} complete.")


def fine_tune_all(net, trainloader, criterion, device):
    print("\nStage 2: Fine-tuning entire MAE ViT...")
    for param in net.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.SGD(net.parameters(), lr=LR_STAGE2, momentum=0.9)
    epoch_stats = []
    for epoch in range(EPOCHS_STAGE2):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_stage2.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/100:.3f}")
                running_loss = 0.0

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        epoch_stats.append((epoch_loss, epoch_acc))
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE2} complete. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Finished Training")
    torch.save(net.state_dict(), WEIGHTS_PATH)
    print("Saved MAE ViT model weights to:", WEIGHTS_PATH)
    print("\nEpoch stats (loss, accuracy) per epoch:")
    print(epoch_stats)


def evaluate(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100*correct/total:.2f}%")

def main():
    # Load datasets
    train_dataset = CustomImageDataset(
        csv_file=TRAIN_CSV_PATH,
        img_dir=DATA_ROOT,
        transform=train_test_transform,
        filename_col=FILENAME_COLUMN,
        label_col=LABEL_COLUMN,
        suffix=IMAGE_SUFFIX,
    )

    test_dataset = CustomImageDataset(
        csv_file=TEST_CSV_PATH,
        img_dir=DATA_ROOT,
        transform=train_test_transform,
        filename_col=FILENAME_COLUMN,
        label_col=LABEL_COLUMN,
        suffix=IMAGE_SUFFIX,
    )

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Load mae-pretrained vit model
    print("Loading MAE-pretrained ViT-B/16 encoder...")
    net = vit_b_16(weights=None)
    mae_ckpt = load_file("/home/jobe/ECS271/project/vit-mae-base/model.safetensors")
    state_dict = mae_ckpt['model'] if 'model' in mae_ckpt else mae_ckpt

    filtered_state_dict = {k.replace("encoder.", ""): v 
                           for k, v in state_dict.items() 
                           if k.startswith("encoder.")}
    
    missing, unexpected = net.load_state_dict(filtered_state_dict, strict=False)
    net.heads.head = nn.Linear(net.heads.head.in_features, NUM_CLASSES)
    print("Loaded MAE ViT-B/16.")

    # Replace classifier head
    net.heads.head = nn.Linear(net.heads.head.in_features, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    loss_weights = torch.tensor([1.0, 4.9, 1.8, 9.4, 6.1], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))

    train_head_only(net, trainloader, criterion, device)
    fine_tune_all(net, trainloader, criterion, device)
    evaluate(net, testloader, device)

if __name__ == "__main__":
    main()
