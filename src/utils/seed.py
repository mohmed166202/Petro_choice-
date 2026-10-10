"""
seed.py - Global Reproducibility & Seed Controller
--------------------------------------------------
Ensures deterministic behavior across Python random, NumPy, PyTorch,
and underlying C libraries / CUDA operations.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Sets global deterministic seed for all libraries.
    
    Args:
        seed (int): The integer random seed (default: 42).
    """
    # 1. Python built-in random and hashing
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 2. NumPy random
    np.random.seed(seed)
    
    # 3. PyTorch (if installed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enforce deterministic algorithms in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


if __name__ == "__main__":
    set_seed(42)
    print("Seed 42 successfully applied across all runtime modules.")
