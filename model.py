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

        self.conv1 = nn.Conv2d(1, 6, 3, padding='same')
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(.3)

    def forward(self, x):
        x = self.transform(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
