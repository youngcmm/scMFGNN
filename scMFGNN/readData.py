from __future__ import print_function, division
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
torch.cuda.current_device()
import h5py
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

datalists = [
             'Plasschaert',
             'Pollen',
             'Quake_10x_Bladder',
             'Quake_10x_Trachea',
             'Quake_Smart-seq2_Heart',
             'Quake_Smart-seq2_Lung',
             'Young',
             'Wang_Lung'
             ]
for data in datalists:
    y = np.loadtxt('data/{}_label.txt'.format(data), dtype=float)
    x = np.loadtxt('data/{}.txt'.format(data), dtype=float)

