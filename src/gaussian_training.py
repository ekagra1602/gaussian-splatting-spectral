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



