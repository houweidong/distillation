import torch.nn.functional as f
import torch.nn as nn
import math
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
# name_t = 'mobilenetv3-large-657e7b3d.pth'
# print('\nloading model from {}'.format(name_t))
# path = os.path.join('/root', '.torch/models/', name_t)
# state_dict = torch.load(path)
# for k in list(state_dict.keys()):
#     if 'conv.5.fc.' not in k:
#         state_dict.pop(k)
# fc0_wm, fc0_bm, fc2_wm, fc2_bm = [], [], [], []
# fc0_wv, fc0_bv, fc2_wv, fc2_bv = [], [], [], []
# for k in state_dict.keys():
#     # print(k, np.mean(state_dict[k].cpu().numpy()), np.var(state_dict[k].cpu().numpy()))
#     if 'fc.0.weight' in k:
#         fc0_wm.append(np.mean(state_dict[k].cpu().numpy()))
#         fc0_wv.append(np.var(state_dict[k].cpu().numpy()))
#     if 'fc.0.bias' in k:
#         fc0_bm.append(np.mean(state_dict[k].cpu().numpy()))
#         fc0_bv.append(np.var(state_dict[k].cpu().numpy()))
#     if 'fc.2.weight' in k:
#         fc2_wm.append(np.mean(state_dict[k].cpu().numpy()))
#         fc2_wv.append(np.var(state_dict[k].cpu().numpy()))
#     if 'fc.2.bias' in k:
#         fc2_bm.append(np.mean(state_dict[k].cpu().numpy()))
#         fc2_bv.append(np.var(state_dict[k].cpu().numpy()))
#
#
# plt.figure()
#
# plt.subplot(2, 4, 1)
# plt.plot(fc0_wm)
# plt.xlabel("fc.0.weight mean")
#
# plt.subplot(2, 4, 2)
# plt.plot(fc0_bm)
# plt.xlabel("fc.0.bias mean")
#
# plt.subplot(2, 4, 3)
# plt.plot(fc2_wm)
# plt.xlabel("fc.2.weight mean")
#
# plt.subplot(2, 4, 4)
# plt.plot(fc2_bm)
# plt.xlabel("fc.2.bias mean")
#
# plt.subplot(2, 4, 5)
# plt.plot(fc0_wv)
# plt.xlabel("fc.0.weight var")
#
# plt.subplot(2, 4, 6)
# plt.plot(fc0_bv)
# plt.xlabel("fc.0.bias var")
#
# plt.subplot(2, 4, 7)
# plt.plot(fc2_wv)
# plt.xlabel("fc.2.weight var")
#
# plt.subplot(2, 4, 8)
# plt.plot(fc2_bv)
# plt.xlabel("fc.2.bias var")
#
# plt.show()


y_true = np.array([0, 0, 1, 1])
y_scores = np.array([500, 100, 1000, 1000])
re = average_precision_score(y_true, y_scores)  # doctest: +ELLIPSIS
print(re)