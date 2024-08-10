import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CharacterDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("L")
        label = int(self.annotations.iloc[idx, 1])

        # Debugging: print the image name and label
        print(f"Loading image: {img_name}, Label: {label}")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


class CharacterRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(CharacterRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        # input dimensions for 32x32 images
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.flatten(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        # return F.log_softmax(x, dim=1)
        return x


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def evaluate(model, validation_loader, criterion):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)
            validation_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(f"Predicted: {pred.cpu().numpy()}, Actual: {target.cpu().numpy()}")


    validation_loss /= len(validation_loader.dataset)
    print(
        f"\nValidation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_loader.dataset)} "
        f"({100. * correct / len(validation_loader.dataset):.0f}%)\n"
    )


if __name__ == "__main__":
    csv_file = "letters-dataset/labels.csv"
    root_dir = "letters-dataset"
    num_classes = 36  # Digits (0-9) and letters (A-Z)

    transform = transforms.Compose(
        [
            # input dimensions for 32x32 images
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = CharacterDataset(
        csv_file=csv_file, root_dir=root_dir, transform=transform
    )

    batch_size = 64
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")


    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler
    )

    model = CharacterRecognitionModel(num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)
        evaluate(model, validation_loader, criterion)
