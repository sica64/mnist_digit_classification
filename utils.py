import torch
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def load_data(data_dir="./data/", batch_size=8):
    # Normalization values found on web
    data_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,), (0.3081,))
    ])

    data = MNIST(data_dir, train=True,
                 download=True, transform=data_transform)
    train_data, val_data = torch.utils.data.random_split(data, [50000, 10000])
    test_data = MNIST(data_dir, train=False,
                      download=True, transform=data_transform)

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader



def get_accuracy(model, dataloader, device="cpu"):
    """
    Calculate accuracy for a given model/dataloader combination.
    """

    img_errors = torch.empty(0, dtype=torch.int64).to(device)
    true_labels = torch.empty(0, dtype=torch.int64).to(device)
    labels_errors = torch.empty(0, dtype=torch.int64).to(device)

    total = 0
    correct = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)

            pred = model(imgs)
            pred_labels = torch.argmax(pred, axis=1)
            img_errors = torch.cat((img_errors, imgs[labels != pred_labels]), 0)
            true_labels = torch.cat((true_labels, labels[labels != pred_labels]), 0)
            labels_errors = torch.cat((labels_errors, pred_labels[labels != pred_labels]), 0)

    return 1 - img_errors.shape[0]/len(dataloader.dataset), img_errors, labels_errors, true_labels
