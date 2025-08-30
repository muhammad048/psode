import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split  # kept for compatibility with commented code
from PIL import Image

# -----------------------------
# Helpers
# -----------------------------
def _using_gpu(args) -> bool:
    """True if CUDA is available and user requested GPUs."""
    try:
        return torch.cuda.is_available() and int(getattr(args, "num_gpu", 0)) > 0
    except Exception:
        return False

def _workers(args) -> int:
    """Read worker count from args (-j / --workers), default to 0 (safe on Windows)."""
    for key in ("workers", "num_workers", "j"):
        if hasattr(args, key):
            try:
                return int(getattr(args, key))
            except Exception:
                pass
    return 0

def _batch_size(args, default=64) -> int:
    try:
        return int(getattr(args, "batch_size", default))
    except Exception:
        return default

# -----------------------------
# Augmentation
# -----------------------------
class Cutout(object):
    """Randomly mask out one or more square patches from a Tensor image (C,H,W)."""
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: Tensor (C, H, W)
        if not torch.is_tensor(img):
            # If mistakenly applied before ToTensor, convert
            img = transforms.ToTensor()(img)
        _, h, w = img.size()
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).to(img.dtype).to(img.device)
        mask = mask.expand_as(img)
        return img * mask

# -----------------------------
# Main CIFAR-100 loader used by training
# -----------------------------
def create_loader_cifar(args):
    """
    Returns: train_loader, val_loader, de_loader
    For DE the code expects a third loader; here we reuse train_loader by default.
    """
    bs = _batch_size(args)
    nw = _workers(args)
    pin = _using_gpu(args)
    # NOTE: persistent_workers must be False on Windows for low-memory stability
    pw = False

    # Train-time transform
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # If you want Cutout on train, enable the next line:
        # Cutout(n_holes=1, length=16),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    # A stronger aug pipeline (unused by default; keep for experiments)
    trans_train_strong = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # apply AFTER ToTensor so it sees (C,H,W)
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    trans_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    root = os.path.join(args.data, "cifar100")
    train_dataset = torchvision.datasets.CIFAR100(root=root, train=True,  transform=trans_train, download=True)
    test_dataset  = torchvision.datasets.CIFAR100(root=root, train=False, transform=trans_val,  download=True)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, drop_last=False
    )
    val_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, drop_last=False
    )

    # For DE evaluation phases many pipelines reuse the train loader; adapt if needed.
    de_loader = train_loader
    return train_loader, val_loader, de_loader

# -----------------------------
# Optional: split CIFAR-100 train set into args.split_num disjoint loaders
# -----------------------------
def split_cifar100(args):
    """
    Splits the CIFAR-100 training set into args.split_num contiguous shards (by iteration order)
    and returns a list of DataLoaders + a single validation loader.
    """
    assert hasattr(args, "split_num") and args.split_num > 0, "args.split_num must be set > 0"
    bs = _batch_size(args)
    nw = _workers(args)
    pin = _using_gpu(args)
    pw = False

    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    trans_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    root = os.path.join(args.data, "cifar100")
    train_dataset = torchvision.datasets.CIFAR100(root=root, train=True,  transform=trans_train, download=True)
    test_dataset  = torchvision.datasets.CIFAR100(root=root, train=False, transform=trans_val,   download=True)

    # Iterate once with batch_size=1 to partition deterministically
    iter_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, drop_last=False
    )

    x_list = [[] for _ in range(args.split_num)]
    y_list = [[] for _ in range(args.split_num)]

    n = len(iter_loader)
    idx_of_dataset = [int(n / args.split_num * i) for i in range(args.split_num + 1)]

    for batch_idx, (inp, target) in enumerate(iter_loader):
        # remove batch dim; now (C,H,W)
        img = inp.squeeze(0)
        lbl = int(target.item()) if torch.is_tensor(target) else int(target)
        # find which shard this index belongs to
        for i in range(1, args.split_num + 1):
            if idx_of_dataset[i - 1] <= batch_idx < idx_of_dataset[i]:
                x_list[i - 1].append(img)
                y_list[i - 1].append(lbl)
                break

    dataset_loaders = [None for _ in range(args.split_num)]
    for i in range(args.split_num):
        images_tensor = torch.stack(x_list[i]) if len(x_list[i]) > 0 else torch.empty(0, 3, 32, 32)
        labels_tensor = torch.tensor(y_list[i], dtype=torch.long) if len(y_list[i]) > 0 else torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(images_tensor, labels_tensor)
        dataset_loaders[i] = DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin, persistent_workers=pw, drop_last=False
        )

    val_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=pw, drop_last=False
    )

    return dataset_loaders, val_loader

# -----------------------------
# Legacy alternative split (kept commented for reference)
# -----------------------------
# def split_cifar100_v2(args):
#     ...
