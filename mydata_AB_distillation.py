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

from models import *
from data.get_data import get_data
from utils.get_tasks import get_tasks
from training.loss_utils import multitask_loss
from ignite.engine import create_supervised_evaluator, _prepare_batch
from training.metric_utils import MultiAttributeMetric
from training.get_loss_metric import get_losses_metrics
from utils.table import print_summar_table
from utils.logger import Logger
from utils.opts import parse_opts
dataset_name = 'new'

parser = argparse.ArgumentParser(description='PyTorch my data Training')
args = parse_opts()
max_epoch = args.max_epoch - args.distill_epoch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Student model
t_net = mobile3l()
if args.network.startswith('mobile3ss'):
    s_net = mobile3ss(pretrained=args.pretrained)
    distill_net = AB_distill_Mobilenetl2Mobilenets(t_net, s_net, args.batch_size, args.DTL, args.loss_multiplier, args.channel_s)
elif args.network.startswith('mobile3s'):
    s_net = mobile3s(pretrained=args.pretrained)
    distill_net = AB_distill_Mobilenetl2Mobilenets(t_net, s_net, args.batch_size, args.DTL, args.loss_multiplier, args.channel_s)
else:
    raise AssertionError("Undefined student network architecture")
if device == 'cpu':
    s_net = torch.nn.DataParallel(s_net).cuda()
    distill_net = torch.nn.DataParallel(distill_net).cuda()
    cudnn.benchmark = True

# Load dataset
attr, attr_name = get_tasks(args)
trainloader, testloader = get_data(args, attr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


criterion_CE, metrics = get_losses_metrics(attr, args.categorical_loss)
train_evaluator = create_supervised_evaluator(s_net, metrics={
    'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)
val_evaluator = create_supervised_evaluator(s_net, metrics={
    'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)
log = Logger(filename=os.path.join(args.log_dir, args.log_file), level='debug')
logger = log.logger.info


# Distillation
def Distillation(distill_net, epoch, withCE=False):
    epoch_start_time = time.time()
    print('\nDistillation Epoch: %d' % epoch)

    distill_net.train()
    distill_net.module.s_net.train()
    distill_net.module.t_net.eval()

    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0

    global optimizer
    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        distill_net.module.batch_size = inputs.shape[0]
        outputs = distill_net(inputs, targets)

        loss = outputs[:, 0].sum()

        if args.DTL is True:
            loss += outputs[:, 2].sum()

        if withCE is True:
            loss += outputs[:, 1].sum()
            # correct += outputs[:, 7].sum().item()
            # total += targets.size(0)

        loss_AT1 = outputs[:, 3].mean()
        loss_AT2 = outputs[:, 4].mean()
        loss_AT3 = outputs[:, 5].mean()
        loss_AT4 = outputs[:, 6].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss1 += loss_AT1.item()
        train_loss2 += loss_AT2.item()
        train_loss3 += loss_AT3.item()
        train_loss4 += loss_AT4.item()

        b_idx = batch_idx

        print('layer1_activation similarity %.1f%%' % (100 * (1 - train_loss1 / (b_idx+1))))
        print('layer2_activation similarity %.1f%%' % (100 * (1 - train_loss2 / (b_idx+1))))
        print('layer3_activation similarity %.1f%%' % (100 * (1 - train_loss3 / (b_idx+1))))
        print('layer4_activation similarity %.1f%%' % (100 * (1 - train_loss4 / (b_idx+1))))

    return train_loss1 / (b_idx+1), train_loss2 / (b_idx+1), train_loss3 / (b_idx+1)


# Training with DTL(Distillation in Transfer Learning) loss
def train_DTL(distill_net, epoch):
    # epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    distill_net.train()
    distill_net.module.s_net.train()
    distill_net.module.t_net.eval()
    train_loss = 0
    global optimizer
    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        distill_net.module.batch_size = inputs.shape[0]
        outputs = distill_net(inputs, targets)

        # CE loss
        loss = outputs[:, 1].sum()

        if args.DTL:
            # DTL loss
            loss += outputs[:, 2].sum()
        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()
        optimizer.step()

        train_loss += loss.item()

        b_idx = batch_idx

    # print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
        print('Loss: %.3f' % (train_loss / (b_idx + 1)))


# Training
def train(net, epoch):
    # epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    global optimizer
    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        net.module.batch_size = inputs.shape[0]
        outputs = net(inputs, targets)
        loss = multitask_loss(outputs, targets, criterion_CE)

        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()
        optimizer.step()

        train_loss += loss.item()
        b_idx = batch_idx

    # print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
        print('Loss: %.3f' % (train_loss / (b_idx + 1)))


# Test
def test(net, epoch):
    net.eval()
    data_list = [trainloader, testloader]
    name_list = ['train', 'val']
    eval_list = [train_evaluator, val_evaluator]

    for data, name, evl in zip(data_list, name_list, eval_list):
        evl.run(data)
        metrics_info = evl.state.metrics["multitask"]
        logger(name + ": Validation Results - Epoch: {}".format(epoch))
        print_summar_table(logger, attr_name, metrics_info['logger'])


# Distillation (Initialization)
optimizer = optim.SGD([{'params': s_net.parameters()},
                       {'params': distill_net.module.Connectors.parameters()}], lr=0.1, nesterov=True, momentum=args.momentum, weight_decay=args.weight_decay)

for epoch in range(1, int(args.distill_epoch) + 1):
    Distillation(distill_net, epoch)

# Cross-entropy training
distill_net.module.stage1 = False
optimizer = optim.SGD([{'params': s_net.parameters()},
                       {'params': distill_net.module.Connectfc.parameters()}], lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
if args.scheduler == 'step':
    optimizer = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.05)
elif args.scheduler == 'cos':
    optimizer = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
elif args.scheduler == 'pleau':
    optimizer = ReduceLROnPlateau(optimizer)
else:
    raise Exception('not implement scheduler')

train_loss = 0
for epoch in range(1, max_epoch+1):
    if args.DTL:
        train_DTL(distill_net, epoch)
    else:
        train(s_net, epoch)

    if epoch % 5 is 0:
        test(s_net, epoch)

    if epoch % 30 is 0:
        save_file_path = os.path.join(args.log_dir, 'save_{}.pth'.format(epoch))
        torch.save(t_net.module.state_dict(), save_file_path)
