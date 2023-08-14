import torch
import os
from h36m_dataset import H36M_Dataset
from extra_dataset import extra_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

dataset_h36m = H36M_Dataset('', 10, 10, 1, split=0)
print(dataset_h36m[0].shape)
dataset_extra = extra_Dataset('', 10, 10, 1)
print(dataset_extra[0].shape)