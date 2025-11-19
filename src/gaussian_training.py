'''
Gaussian Splatting Training Script

Custom loss function for Gaussian Splatting training.

'''

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import imageio

from datasets.colmap import Dataset as ColmapDataset, Parser
from utils import rgb_to_sh, knn as knn_sklearn

optimizers = {
    "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * lr_scale, eps=1e-15),  # Only means LR is scaled!
    "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
    "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
    "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2, eps=1e-15),
    "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3, eps=1e-15),
    "shN": torch.optim.Adam([params["shN"]], lr=2.5e-3 / 20, eps=1e-15),
}
return optimizers

