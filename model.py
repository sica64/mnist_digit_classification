import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


class DigitClassifierCNN(nn.Module):

    def __init__(self):
        super(DigitClassifierCNN, self).__init__()

        # reshape input into 28x28 black and white image.
        self.transform = v2.Compose([
            v2.Resize([28, 28], antialias=True),
            v2.Grayscale(),
        ])

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 10)

        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.transform(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)

        return x
