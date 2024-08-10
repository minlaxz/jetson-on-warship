from torch import nn
from torch.nn import functional as F


# class CharacterRecognitionModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CharacterRecognitionModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(0.25)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.flatten = nn.Flatten()
#         # Adjusted input dimensions for 64x64 images
#         self.fc1 = nn.Linear(128 * 6 * 6, 256)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.dropout_fc = nn.Dropout(0.5)

#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.dropout(x)

#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.dropout(x)

#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.dropout(x)

#         x = self.flatten(x)
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = self.dropout_fc(x)
#         x = self.fc2(x)

#         return F.log_softmax(x, dim=1)


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

        return F.log_softmax(x, dim=1)


# class CharacterRecognitionModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CharacterRecognitionModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(0.25)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.flatten = nn.Flatten()
#         # Adjust the input size to 512 (128 * 2 * 2)
#         self.fc1 = nn.Linear(128 * 2 * 2, 256)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.dropout_fc = nn.Dropout(0.5)

#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.dropout(x)

#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.dropout(x)

#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.dropout(x)

#         x = self.flatten(x)
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = self.dropout_fc(x)
#         x = self.fc2(x)

#         return F.log_softmax(x, dim=1)
