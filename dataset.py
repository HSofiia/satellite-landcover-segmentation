import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SatelliteDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return image, mask


def get_dataloaders(batch_size=8):

    images = np.load("data/filtered_images.npy")
    masks = np.load("data/filtered_masks.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    train_dataset = SatelliteDataset(X_train, y_train)
    val_dataset = SatelliteDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader