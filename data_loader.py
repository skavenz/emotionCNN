import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_data(batch_size):
    current_dir = os.getcwd()
    train_dir = os.path.join(current_dir, "dataset", "train")
    test_dir = os.path.join(current_dir, "dataset", "test")

    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])

    train_set = ImageFolder(train_dir, transform=preprocess)
    test_set = ImageFolder(test_dir, transform=preprocess)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
