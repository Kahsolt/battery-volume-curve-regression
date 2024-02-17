#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/14

from pathlib import Path
import pickle as pkl
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Model
from torch import Tensor
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
SUBMIT_PATH = LOG_PATH / 'submit.csv'

# 一个循环: 恒流转恒压充电 -> 静置 -> 恒流放电 -> 静置 (1-0-2-0)
N_ACTION_CYCLE = 4
ACTION_STATUS = {
  '静置': 0,
  '恒流转恒压充电': 1,
  '恒流放电': 2,
}
N_WIN = 5
N_PREC = 3

SEED = 114514
