'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import time
import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from ignite.engine import _prepare_batch
from models import *
from data.get_data import get_data
from utils.get_tasks import get_tasks
from training.loss_utils import multitask_loss
from ignite.engine import create_supervised_evaluator
from training.metric_utils import MultiAttributeMetric
from training.get_loss_metric import get_losses_metrics
from utils.table import print_summar_table
from utils.logger import Logger
from utils.opts import parse_opts


# Training
def train(net, epoch):
    # epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    net.train()
    # train_loss = 0
    global optimizer
    bt_sum = len(trainloader)
    print('lr: %.4f' % optimizer.optimizer.param_groups[0]['lr'])
    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        outputs = net(inputs)
        loss = multitask_loss(outputs, targets, criterion_CE)

        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()
        # train_loss += loss.item()
        # print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
        if batch_idx % 20 == 0:
            print('Loss: %.3f[%d/%d] ' % (loss.item(), batch_idx, bt_sum))


# Test
def test(net, epoch):
    net.eval()
    data_list = [trainloader, testloader]
    # name_list = ['train', 'val']
    name_list = ['val']
    eval_list = [train_evaluator, val_evaluator]

    for data, name, evl in zip(data_list, name_list, eval_list):
        evl.run(data)
        metrics_info = evl.state.metrics["multitask"]
        # logger(name + ": Validation Results - Epoch: {}".format(epoch))
        # print_summar_table(logger, attr_name, metrics_info['logger'])
    if args.scheduler == 'pleau':
        optimizer.step(metrics_info['logger']['attr']['ap'][-1])
    else:
        optimizer.step()
    return metrics_info


class Saver():
    def __init__(self):
        self.max_ap = 0.0
        self.save_root = args.log_dir

    def save(self, epoch, metrics_info):
        ap = metrics_info['logger']['attr']['ap'][-1]
        if epoch > 15 and ap > self.max_ap:
            self.max_ap = ap
            save_file_path = os.path.join(self.save_root, 'ap{}'.format(ap))
            torch.save(t_net.module.state_dict(), save_file_path)

            logger(": Validation Results - Epoch: {}".format(epoch))
            print_summar_table(logger, attr_name, metrics_info['logger'])
            logger('AP:%0.3f' % metrics_info['logger']['attr']['ap'][-1])


parser = argparse.ArgumentParser(description='PyTorch my data Training')
args = parse_opts()
log = Logger(filename=os.path.join(args.log_dir, args.log_file), level='debug')
logger = log.logger.info
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model, dataloader, opmiter, use ignite's evaluator
t_net = mobile3l(frm='official', )
if device == 'cuda':
    t_net = torch.nn.DataParallel(t_net).cuda()

attr, attr_name = get_tasks(args)
criterion_CE, metrics = get_losses_metrics(attr, args.categorical_loss)
trainloader, testloader = get_data(args, attr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

optimizer = optim.SGD(t_net.parameters(), lr=args.lr, nesterov=args.nesterov, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.adam
if args.scheduler == 'step':
    optimizer = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
elif args.scheduler == 'cos':
    optimizer = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-3)
elif args.scheduler == 'pleau':
    optimizer = ReduceLROnPlateau(optimizer)
else:
    raise Exception('not implement scheduler')

train_evaluator = create_supervised_evaluator(t_net, metrics={
    'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)
val_evaluator = create_supervised_evaluator(t_net, metrics={
    'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)

Saver = Saver()


# Distillation (Initialization)
for epoch in range(1, args.n_epochs+1):
    train(t_net, epoch)
    metric_info = test(t_net, epoch)
    Saver.save(epoch, metric_info)
