"""
Some utils functions
"""

import torch
import numpy as np
import random


def make_deterministic(seed):
    # Making Pytorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Making Python deterministic
    random.seed(seed)
    # Making numpy deterministic
    np.random.seed(seed)