import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def calc_output_shape(x, kernel, padding=0, stride=1):
    w = x.shape[0]
    w = int((w - kernel + 2*padding) / stride) + 1

    return w, w, x.shape[2]


class DigitClassifierCNN(nn.Module):

    def __init__(self, apply_batch_norm=False, apply_dropout=False,
                 ch1=16, kernel=3,
                 l1=128, l2=32):
        super(DigitClassifierCNN, self).__init__()

        ch2 = ch1*2

        # reshape input into 28x28 black and white image.
        self.transform = v2.Compose([
            v2.Resize([28, 28], antialias=True),
            v2.Grayscale(),
        ])

        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout = apply_dropout

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, ch1, kernel)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel)
        self.bn2 = nn.BatchNorm2d(ch2)

        # calculate sizes after conv layer and max pooling.
        conv1_output_size = (28 - kernel + 1) // 2
        conv2_output_size = (conv1_output_size - kernel + 1) // 2

        self.fc1 = nn.Linear(int(conv2_output_size)**2 * ch2, l1)
        self.bn3 = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(l1, l2)
        self.bn4 = nn.BatchNorm1d(l2)
        self.fc3 = nn.Linear(l2, 10)

        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.transform(x)

        if self.apply_batch_norm:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = torch.flatten(x, 1)
            if self.apply_dropout:
                x = self.dropout(x)

            x = F.relu(self.bn3(self.fc1(x)))
            x = F.relu(self.bn4(self.fc2(x)))

            if self.apply_dropout:
                x = self.dropout(x)

        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            if self.apply_dropout:
                x = self.dropout(x)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            if self.apply_dropout:
                x = self.dropout(x)

        x = self.fc3(x)

        return x
