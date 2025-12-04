import os
import glob
from PIL import Image

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torchvision.transforms as T

from autoencoder import AutoEncoder, AEClassifier

class APTOSDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        label = row['diagnosis']

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def train_autoencoder(model, loader, epochs=10, lr=1e-3, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, _ in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            recon = model(imgs)
            loss = criterion(recon, imgs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        if save_path is not None:
            torch.save(model.state_dict(), f"{save_path}_epoch{epoch}.pth")

        print(f"AE Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(loader.dataset):.4f}")

def train_classifier(model, loader, epochs=10, lr=1e-3, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total, correct, epoch_loss = 0, 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            _, predicted = preds.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"CLS Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/total:.4f} | Acc: {acc:.2f}%")

        if save_path is not None:
            torch.save(model.state_dict(), f"{save_path}_epoch{epoch}.pth")

def evaluate(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            _, predicted = preds.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def evaluate_autoencoder(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

def main():
    print("beginning training")
    BASE_PATH = "/home/jobe/ECS271/project/"
    CSV_PATH = f"{BASE_PATH}dataset/aptos/train.csv"
    IMG_DIR  = f"{BASE_PATH}dataset/aptos/train_images"
    SAVE_DIR = f"{BASE_PATH}weights"
    os.makedirs(SAVE_DIR, exist_ok=True)

    ae_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    cls_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
    ])

    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["diagnosis"]   # ensures class balance
    )

    train_df.to_csv(f"{BASE_PATH}dataset/aptos/train_split.csv", index=False)
    test_df.to_csv(f"{BASE_PATH}dataset/aptos/test_split.csv", index=False)

    # Datasets
    ae_train_ds = APTOSDataset(f"{BASE_PATH}dataset/aptos/train_split.csv", IMG_DIR, transform=ae_transform)
    cls_train_ds = APTOSDataset(f"{BASE_PATH}dataset/aptos/train_split.csv", IMG_DIR, transform=cls_transform)

    ae_test_ds = APTOSDataset(f"{BASE_PATH}dataset/aptos/test_split.csv", IMG_DIR, transform=ae_transform)
    cls_test_ds = APTOSDataset(f"{BASE_PATH}dataset/aptos/test_split.csv", IMG_DIR, transform=cls_transform)

    # Dataloaders
    ae_train_loader = DataLoader(ae_train_ds, batch_size=32, shuffle=True)
    ae_test_loader = DataLoader(ae_test_ds, batch_size=32, shuffle=False)

    cls_train_loader = DataLoader(cls_train_ds, batch_size=32, shuffle=True)
    cls_test_loader = DataLoader(cls_test_ds, batch_size=32, shuffle=False)

    # Pretrain Autoencoder
    autoenc = AutoEncoder()
    ae_path = f"{SAVE_DIR}/autoencoder"
    cls_path = f"{SAVE_DIR}/classifier"
    if os.path.exists(f"{ae_path}_epoch10.pth"):
        print("Loading pretrained autoencoder (epoch10)...")
        autoenc.load_state_dict(torch.load(f"{ae_path}_epoch10.pth"))
    else:
        print("Pretraining autoencoder...")
        train_autoencoder(
            autoenc, ae_train_loader,
            epochs=10,
            lr=1e-3,
            save_path=ae_path   # <— saves autoencoder_epochX.pth
        )

    test_loss = evaluate_autoencoder(autoenc, ae_test_loader)
    print("AE Validation Loss:", test_loss)

    # Train Classifier
    classifier = AEClassifier(autoenc.encoder)
    if os.path.exists(f"{cls_path}_epoch25.pth"):
        print("Loading pretrained classifier (epoch20)...")
        classifier.load_state_dict(torch.load(f"{cls_path}_epoch20.pth"))
    else:
        print("Training classifier...")
        train_classifier(
            classifier, cls_train_loader,
            epochs=25,
            lr=1e-4,
            save_path=cls_path  # <— saves classifier_epochX.pth
        )

    test_acc = evaluate(classifier, cls_test_loader)
    print("Validation Accuracy:", test_acc)

if __name__ == "__main__":
    main()


