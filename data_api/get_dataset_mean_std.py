from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms


def get_mean_std(dataset=None, ratio=0.1, mean_std_path=None):
    """
    dataset:

    """
    base_path = os.path.dirname(__file__)
    if not mean_std_path or dataset is not None:
        assert dataset is not None
        loader = DataLoader(dataset, batch_size=int(len(dataset)*ratio), num_workers=0, shuffle=False, drop_last=True)
        train = next(iter(loader))[0]
        mean = np.mean(train.numpy(), axis=(0, 2, 3))
        std = np.std(train.numpy(), axis=(0, 2, 3))
        np.savez(os.path.join(base_path, "mean_std.npz"), mean=mean, std=std)
    else:
        data = np.load(os.path.join(base_path, "mean_std.npz"))
        mean = data["mean"]
        std = data["std"]

    return transforms.Normalize(mean=mean, std=std), [mean, std]



