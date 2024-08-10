import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms


# Custom dataset class
class CharacterDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("L") # Convert image to grayscale, called Luminescence
        label = self.annotations.iloc[idx, 1]
        print(label)
        # Map 'A'-'Z' to 0-25, and '0'-'9' to 26-35
        if label.isalpha():  # Check if the label is a letter
            label = ord(label.upper()) - ord('A')  # 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
        elif label.isdigit():  # Check if the label is a digit
            label = ord(label) - ord('0') + 26  # '0' -> 26, '1' -> 27, ..., '9' -> 35

        print(label)
        # Convert the numerical label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        print(label)

        if self.transform:
            image = self.transform(image)

        return image, label


csv_file = "letters-dataset/labels.csv"
root_dir = "letters-dataset"

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
