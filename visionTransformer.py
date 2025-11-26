# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful

Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
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

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# !!! CUSTOM DATASET PATHS AND FILES !!!
# Replace these with the paths to your data root directories and CSV files.
DATA_ROOT = './data/aptos2019-blindness-detection/train_images' # Directory with all images
TRAIN_CSV_PATH = './data/aptos2019-blindness-detection/train.csv'
TEST_CSV_PATH = './data/aptos2019-blindness-detection/train.csv'
FILENAME_COLUMN = 'id_code' # Column in your CSV that holds the hex image ID
LABEL_COLUMN = 'diagnosis'    # Column in your CSV that holds the class name
IMAGE_SUFFIX = '.png'        # File extension for your images

class CustomImageDataset(Dataset):
    """
    A custom Dataset class to handle image classification where all images 
    are in one directory and labels are provided in a separate CSV file.
    
    The CSV file is expected to have columns for the image filename/ID and the class label.
    Example CSV format:
    id,label
    a3b4c5.jpg,cat
    d6e7f8.jpg,dog
    ...
    """
    def __init__(self, csv_file, img_dir, transform=None, filename_col='id_code', label_col='diagnosis', suffix='.png'):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            filename_col (str): Name of the column containing image IDs/filenames.
            label_col (str): Name of the column containing class labels.
            suffix (str): File extension to append to the ID (e.g., '.jpg').
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.filename_col = filename_col
        self.label_col = label_col
        self.suffix = suffix
        
        # Create a mapping from string class names to integer indices
        self.classes = sorted(self.annotations[label_col].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Dataset initialized with {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 1. Get the image ID and label from the CSV row
        row = self.annotations.iloc[idx]
        img_id = str(row[self.filename_col])
        label_name = row[self.label_col]
        
        # 2. Construct the full path and load the image
        img_path = os.path.join(self.img_dir, img_id + self.suffix)
        
        # Ensure the image file exists (critical for debugging file issues)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        image = Image.open(img_path).convert('RGB') # Load and ensure 3 color channels

        # 3. Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # 4. Convert string label to numerical index
        label = self.class_to_idx[label_name]
        
        return image, label

if __name__ == '__main__':
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_CLASSES = 5
    
    train_and_test_transform = transforms.Compose([
    transforms.Resize(256), # Resize smallest side to 256
    transforms.CenterCrop(IMAGE_SIZE), # Take a 224x224 center crop
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # Load custom datasets using the new CustomImageDataset class
    print(f"\nAttempting to load data using custom CSV/Folder structure...")
    try:
        # Use the same image directory for train and test if your files are split via CSV
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
        # Fallback to CIFAR10 for demonstration if custom path is not found
        print("WARNING: Custom data paths or CSV files not found. Falling back to CIFAR10 for runnable demo.")
        print(f"Error: {e}")
        print("Please replace DATA_ROOT, TRAIN_CSV_PATH, and TEST_CSV_PATH with your actual paths.")


    # Dynamically determine the number of classes from the loaded dataset
    # We rely on the .classes attribute provided by both CustomImageDataset and CIFAR10/ImageFolder
    print(f"\nModel will be trained on {NUM_CLASSES} classes.")

    # Create DataLoaders for batching
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    classes = (0, 1, 2, 3, 4)

    net = ViT(
        image_size = IMAGE_SIZE,
        patch_size = 16,
        num_classes = NUM_CLASSES,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print("Training batch ", i)
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # outputs = net(images)

    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                             for j in range(4)))

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1


    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))