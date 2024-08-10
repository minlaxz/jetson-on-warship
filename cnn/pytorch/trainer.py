import warnings

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# Check if CUDA (Nvidia GPU) is available, otherwise use CPU
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("mps")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")
print(f"Using device: {device}")

# Set the seed for reproducibility
np.random.seed(2)
torch.manual_seed(2)

# Label mapping
label_number = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "I": 18,
    "J": 19,
    "K": 20,
    "L": 21,
    "M": 22,
    "N": 23,
    "P": 24,
    "Q": 25,
    "R": 26,
    "S": 27,
    "T": 28,
    "U": 29,
    "V": 30,
    "W": 31,
    "X": 32,
    "Y": 33,
    "Z": 34,
}

label_word = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

# Prepare the dataset dataframe
dataframed = pd.DataFrame(columns=["path", "label"])
for x in range(0, len(label_word)):
    temp = "./letters-dataset/" + label_word[x] + "/"
    for dirname, _, filenames in os.walk(temp):
        for filename in filenames:
            name = filename
            label = label_number[label_word[x]]
            dataframed.loc[len(dataframed)] = [temp + "/" + name, label]

print(dataframed.head())
print("Shape of the Dataset = ", dataframed.shape)

# Split the data into train, test, and validation sets
train, test = train_test_split(dataframed, test_size=0.2, random_state=42)
test, valid = train_test_split(test, test_size=0.5, random_state=42)
print(train.head())
print(test.head())
print(valid.head())


# Custom Dataset class
class CharacterDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))

        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


# Transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
)

# Create datasets and dataloaders
train_dataset = CharacterDataset(train, transform=transform)
valid_dataset = CharacterDataset(valid, transform=transform)
test_dataset = CharacterDataset(test, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 35)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Instantiate the model, loss function, and optimizer
model = CNNModel().to(device)  # Move the model to MPS if available
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.3, min_lr=1e-6
)


# Training function
def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=10,
):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = (
                images.to(device),
                labels.to(device),
            )  # Move images and labels to MPS if available
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        val_accuracy = evaluate_model(model, valid_loader)

        scheduler.step(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%, Validation Accuracy: {val_accuracy}%"
        )


# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = (
                images.to(device),
                labels.to(device),
            )  # Move images and labels to MPS if available
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device)

# Test the model
test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy}%")


# Prediction visualization
def visualize_predictions(model, test_loader, label_word, num_images=50):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = (
        images.to(device),
        labels.to(device),
    )  # Move images and labels to MPS if available
    outputs = model(images)
    _, predicted = outputs.max(1)

    for i in range(num_images):
        plt.figure()
        plt.imshow(
            images[i].cpu().squeeze(), cmap="gray"
        )  # Move back to CPU for visualization
        plt.title(
            f"Predicted: {label_word[predicted[i].item()]} | Actual: {label_word[labels[i].item()]}"
        )
        plt.show()


# visualize_predictions(model, test_loader, label_word, device)
