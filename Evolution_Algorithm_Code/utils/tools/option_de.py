# Evolution_Algorithm_Code/utils/tools/option_de.py
import argparse
import os
import random
from typing import Tuple, Optional
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ----------------------------
# AMP compatibility wrapper
# ----------------------------
# Use torch.amp.autocast('cuda') if available, else fall back to torch.cuda.amp.autocast()
try:
    from torch.amp import autocast as _torch_amp_autocast

    def amp_autocast():
        return _torch_amp_autocast('cuda')
except Exception:
    try:
        from torch.cuda.amp import autocast as _cuda_amp_autocast

        def amp_autocast():
            return _cuda_amp_autocast()
    except Exception:
        @contextmanager
        def amp_autocast():
            yield  # no-op if AMP isn't available


# ----------------------------
# Argument parser
# ----------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Options for DE/PSO + CADE pipeline")

    # Data / IO
    parser.add_argument('--data', default='./data', type=str,
                        help='dataset root directory (download target if missing)')
    parser.add_argument('--dataset', default=None, choices=[None, 'cifar10', 'cifar100'],
                        help='explicit dataset selection; if None, inferred from --num_classes')
    parser.add_argument('--num_classes', default=100, type=int,
                        help='number of classes (10 → CIFAR-10, 100 → CIFAR-100)')
    parser.add_argument('--output', default='./cade_results', type=str,
                        help='output directory for artifacts')
    parser.add_argument('--log_dir', default='./cade_results/log.txt', type=str,
                        help='path to a log file (will be created if missing)')
    parser.add_argument('--exp_name', default='exp_debug', type=str,
                        help='experiment tag for output dir names')
    parser.add_argument('--model', default='SEW_resnet34', type=str,
                        help='label used in saved filenames')

    # Compute / performance
    parser.add_argument('--num-gpu', default=1, type=int, metavar='N',
                        help='number of GPUs to use')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='data loading workers for train/DE')
    parser.add_argument('-j_val', '--workers_val', default=2, type=int, metavar='N',
                        help='data loading workers for validation/eval')
    parser.add_argument('--amp', action='store_true',
                        help='enable mixed precision (AMP)')
    parser.add_argument('--seed', default=42, type=int, help='random seed (<=0 to disable)')

    # Distributed toggles (safe defaults even if you don't use DDP)
    parser.add_argument('--rank', default=0, type=int, help='global rank (DDP)')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank (DDP)')
    parser.add_argument('--world_size', default=1, type=int, help='world size (DDP)')
    parser.add_argument('--distributed', action='store_true', help='enable DDP mode')

    # Evolutionary search
    parser.add_argument('--popsize', default=20, type=int, help='population size')
    parser.add_argument('--de_epochs', default=50, type=int,
                        help='number of DE generations (use range(args.de_epochs))')
    parser.add_argument('--de_batch_size', default=128, type=int,
                        help='mini-batch size used for DE/fitness evaluation')
    parser.add_argument('--test_batch_size', default=256, type=int,
                        help='mini-batch size used for validation/testing')
    parser.add_argument('--de_slice_len', default=0, type=int,
                        help='>0 to cap DE eval batches; 0 uses full loader')
    parser.add_argument('--pop_init', default=None, type=str,
                        help='directory with ≥popsize checkpoints to seed initial population')
    parser.add_argument('--f_init', default=0.5, type=float, help='initial DE mutation factor F')
    parser.add_argument('--cr_init', default=0.9, type=float, help='initial DE crossover rate CR')

    # PSO controller for (F, CR)
    parser.add_argument('--use_pso', action='store_true',
                        help='run a short PSO to select (F, CR) before DE')
    parser.add_argument('--pso_popsize', default=8, type=int, help='PSO swarm size')
    parser.add_argument('--pso_iters', default=10, type=int, help='PSO iterations')
    parser.add_argument('--pso_eval_gens', default=3, type=int,
                        help='cheap DE generations per PSO fitness evaluation')

    # Misc
    parser.add_argument('--test_ood', action='store_true', help='run OOD code paths if present')

    return parser


parser = _build_parser()
# Parse at import time to keep the original project pattern:
args = parser.parse_args()


# args_text (string) expected by main_cosde.py (simple YAML-like dump)
def _format_args(a) -> str:
    lines = []
    for k, v in sorted(vars(a).items()):
        lines.append(f"{k}: {v}")
    return "\n".join(lines)
args_text = _format_args(args)


# ----------------------------
# Utilities
# ----------------------------
def _set_seed(seed: int):
    if seed and seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _infer_dataset_name(num_classes: int) -> str:
    return 'cifar10' if int(num_classes) == 10 else 'cifar100'


def _cifar_stats(name: str) -> Tuple[list, list]:
    if name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2470, 0.2435, 0.2616]
    else:
        # CIFAR-100
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]
    return mean, std


def _build_transforms(name: str):
    mean, std = _cifar_stats(name)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def _build_cifar_loaders(name: str,
                         root: str,
                         train_bs: int,
                         test_bs: int,
                         workers_train: int,
                         workers_val: int):
    train_tf, eval_tf = _build_transforms(name)
    os.makedirs(root, exist_ok=True)

    if name == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=root, train=True,  transform=train_tf, download=True)
        val_set   = torchvision.datasets.CIFAR10(root=root, train=False, transform=eval_tf,  download=True)
        classes = 10
    else:
        train_set = torchvision.datasets.CIFAR100(root=root, train=True,  transform=train_tf, download=True)
        val_set   = torchvision.datasets.CIFAR100(root=root, train=False, transform=eval_tf,  download=True)
        classes = 100

    train_loader = DataLoader(
        train_set, batch_size=train_bs, shuffle=True,
        num_workers=max(0, workers_train), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=test_bs, shuffle=False,
        num_workers=max(0, workers_val), pin_memory=True, drop_last=False
    )
    # For DE/PSO fitness, we typically use the train (or a subset) loader
    de_loader = train_loader
    return classes, train_loader, val_loader, de_loader


# ----------------------------
# Loader entrypoint
# ----------------------------
def obtain_loader(a=args):
    """
    Returns (loader_train, loader_eval, loader_de)

    - loader_train: training loader (may be None if your pipeline does not train per-epoch)
    - loader_eval : evaluation/validation loader (used for scoring accuracy/precision)
    - loader_de   : loader used during DE/PSO fitness evaluations (usually the train loader)
    """
    # --- Prevent UnboundLocalError by initializing locals
    loader_train: Optional[DataLoader] = None
    loader_eval: Optional[DataLoader] = None
    loader_de: Optional[DataLoader] = None

    # Seed everything (if enabled)
    _set_seed(getattr(a, 'seed', 0))

    # Decide dataset
    dataset_name = getattr(a, 'dataset', None)
    if dataset_name is None:
        dataset_name = _infer_dataset_name(getattr(a, 'num_classes', 100))
        setattr(a, 'dataset', dataset_name)

    # Reconcile args.num_classes with dataset, warn if mismatch
    expected_classes = 10 if dataset_name == 'cifar10' else 100
    if int(getattr(a, 'num_classes', expected_classes)) != expected_classes:
        print(f"[warn] Overriding --num_classes={a.num_classes} to {expected_classes} for {dataset_name}.")
        a.num_classes = expected_classes

    # Build loaders
    if dataset_name in ('cifar10', 'cifar100'):
        classes, tr_loader, va_loader, de_loader = _build_cifar_loaders(
            name=dataset_name,
            root=getattr(a, 'data', './data'),
            train_bs=getattr(a, 'de_batch_size', 128),
            test_bs=getattr(a, 'test_batch_size', 256),
            workers_train=getattr(a, 'workers', 2),
            workers_val=getattr(a, 'workers_val', 2),
        )
        # expose resolved class-count just in case
        a.num_classes = classes

        loader_train = tr_loader
        loader_eval  = va_loader
        loader_de    = de_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Final sanity
    if loader_eval is None or loader_de is None:
        raise RuntimeError(f"Failed to build loaders for dataset={dataset_name} "
                           f"(loader_eval={loader_eval}, loader_de={loader_de}).")
    return loader_train, loader_eval, loader_de


__all__ = ["args", "args_text", "amp_autocast", "obtain_loader", "parser"]
