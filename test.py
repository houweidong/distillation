import torch.nn.functional as f
import torch.nn as nn
import math
import torch
import numpy as np


path = '/root/.torch/models/ap0.8972'
model_dict = torch.load(path, map_location='cpu')

for k, v in model_dict.items():
    if k.startswith('class'):
        print(k)
# print(model)