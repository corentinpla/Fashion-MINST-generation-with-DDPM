import math
from inspect import isfunction
from functools import partial
import random
import IPython
import pandas as pd
from datasets import load_dataset, load_from_disk

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torch.optim import Adam

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt

import functions


# Reproductibility
torch.manual_seed(53)
random.seed(53)
np.random.seed(53)

batch_size = 64
data_path = "fashion_mnist"  