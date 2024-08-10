import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from dataset import CharacterDataset, transform, csv_file, root_dir
from model import CharacterRecognitionModel

dataset = CharacterDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)


# Data Loaders
batch_size = 64
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(validation_split * dataset_size)
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data loaders
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
# End of Data Loaders


# Example usage
# Digits (0-9) and letters (A-Z)
num_classes = 36
model = CharacterRecognitionModel(num_classes)


# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
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


# Evaluation loop
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

    validation_loss /= len(validation_loader.dataset)
    print(
        f"\nValidation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(validation_loader.dataset)} "
        f"({100. * correct / len(validation_loader.dataset):.0f}%)\n"
    )


# Run training and evaluation
num_epochs = 1
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}")
    train(model, train_loader, criterion, optimizer, epoch)
    evaluate(model, validation_loader, criterion)

# Save the model
# torch.save(model.state_dict(), "character_recognition_model.pth")
