import torch
from torch.utils.data import DataLoader
from m_autoencoder_vit import CustomImageDataset, train_test_transform  
from MAEVitModel import MyViTModel

BATCH_SIZE = 32
NUM_CLASSES = 5
FILENAME_COLUMN = "id_code"
LABEL_COLUMN    = "diagnosis"
IMAGE_SUFFIX    = ".png"
TRAIN_CSV_PATH = "/home/jobe/ECS271/project/dataset/aptos/train_split.csv" 
TEST_CSV_PATH  = "/home/jobe/ECS271/project/dataset/aptos/test_split.csv"
DATA_ROOT      = "/home/jobe/ECS271/project/dataset/aptos/train_images"
WEIGHTS_PATH   = "/home/jobe/ECS271/project/weights/mae_vit.safetensors" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(net, testloader, device):
    net.eval()
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
    test_dataset = CustomImageDataset(
        csv_file=TEST_CSV_PATH,
        img_dir=DATA_ROOT,
        transform=train_test_transform,
        filename_col=FILENAME_COLUMN,
        label_col=LABEL_COLUMN,
        suffix=IMAGE_SUFFIX,
    )

    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    net = MyViTModel(num_classes=5).to(DEVICE)
    net.load_trained_weights(WEIGHTS_PATH, device=DEVICE)

    evaluate(net, testloader, DEVICE)

if __name__ == "__main__":
    main()