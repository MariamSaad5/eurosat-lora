import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG we use so a given config produces reproducible splits,
    weight initialization, and data ordering."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
