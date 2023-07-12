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


def get_dataset(data_path, batch_size, test = False):
    
    dataset = load_dataset(data_path)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),  # Transform PIL image into tensor of value between [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize values between [-1,1], plus pratique pour entrainer
    ])

    # define function for HF dataset transform


    def transforms_im(examples):
        examples['pixel_values'] = [transform(image) for image in examples['image']]
        del examples['image']
        return examples

    dataset = dataset.with_transform(transforms_im).remove_columns('label')  # We don't need it 
    channels, image_size, _ = dataset['train'][0]['pixel_values'].shape
        
    if test:
        dataloader = DataLoader(dataset['test'], batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

    len_dataloader = len(dataloader)
    print(f"channels: {channels}, image dimension: {image_size}, len_dataloader: {len_dataloader}")  
    
    return dataloader, channels, image_size, len_dataloader