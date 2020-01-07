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
from utils.log_config import log_config


# Distillation
def Distillation(distill_net, epoch, withCE=False):
    logger('\nDistillation Epoch: %d  LR: %.4f' % (epoch, optimizer.optimizer.param_groups[0]['lr']))

    distill_net.train()
    distill_net.module.s_net.train()
    distill_net.module.t_net.eval()

    train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = 0, 0, 0, 0, 0

    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        distill_net.module.batch_size = inputs.shape[0]
        outputs = distill_net(inputs, targets)
        bt_sum = len(trainloader)
        loss = outputs[:, 0].sum()
        loss_AT = loss.item()

        if args.DTL is True:
            loss1 = outputs[:, 2].sum()
            loss += loss1
            loss_DTL = loss1.item()
        if withCE is True:
            loss += outputs[:, 1].sum()

        loss_AT1, loss_AT2, loss_AT3, loss_AT4 = outputs[:, 3].mean(), outputs[:, 4].mean(), outputs[:, 5].mean(), outputs[:, 6].mean()

        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()

        train_loss += loss.item()
        train_loss1 += loss_AT1.item()
        train_loss2 += loss_AT2.item()
        train_loss3 += loss_AT3.item()
        train_loss4 += loss_AT4.item()

        similarity1 = 100 * (1 - train_loss1 / (batch_idx+1))
        similarity2 = 100 * (1 - train_loss2 / (batch_idx+1))
        similarity3 = 100 * (1 - train_loss3 / (batch_idx+1))
        similarity4 = 100 * (1 - train_loss4 / (batch_idx+1))
        if batch_idx % 20 == 0:
            logger('similarity1: %.1f  similarity2: %.1f  similarity3: %.1f  similarity4: %.1f  loss_AT: %.3f  loss_DTL: %.3f[%d/%d]'
                  % (similarity1, similarity2, similarity3, similarity4, loss_AT, loss_DTL, batch_idx, bt_sum))

    optimizer.step()

# Training with DTL(Distillation in Transfer Learning) loss
def train_DTL(distill_net, epoch):
    logger('\nClassification training Epoch: %d  LR: %.4f' % (epoch, optimizer.optimizer.param_groups[0]['lr']))
    distill_net.train()
    distill_net.module.s_net.train()
    distill_net.module.t_net.eval()
    bt_sum = len(trainloader)
    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        distill_net.module.batch_size = inputs.shape[0]
        outputs = distill_net(inputs, targets)

        # CE loss
        loss = outputs[:, 1].sum()
        loss_CE = loss.item()

        if args.DTL:
            # DTL loss
            loss1 = outputs[:, 2].sum()
            loss += loss1
            loss_DTL = loss1.item()
        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()

        if batch_idx % 20 == 0:
            logger('Loss: %.3f Loss_CE: %.3f Loss_DTL %.3f[%d/%d] ' % (loss.item(), loss_CE, loss_DTL, batch_idx, bt_sum))


# Training
def train(net, epoch):
    # epoch_start_time = time.time()
    logger('\nClassification training Epoch: %d  LR: %.4f' % (epoch, optimizer.optimizer.param_groups[0]['lr']))
    net.train()
    bt_sum = len(trainloader)
    for batch_idx, bt in enumerate(trainloader):
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        net.module.batch_size = inputs.shape[0]
        outputs = net(inputs)
        loss = multitask_loss(outputs, targets, criterion_CE)

        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()

        if batch_idx % 20 == 0:
            logger('Loss: %.3f[%d/%d] ' % (loss.item(), batch_idx, bt_sum))


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

    if args.scheduler == 'pleau':
        optimizer.step(metrics_info['logger']['attr']['ap'][-1])
    else:
        optimizer.step()
    return metrics_info


class Saver(object):
    def __init__(self):
        self.max_ap = 0.0
        self.save_root = args.log_dir

    def save(self, epoch, metrics_info):
        ap = metrics_info['logger']['attr']['ap'][-1]
        if ap > self.max_ap:
            self.max_ap = ap
            save_file_path = os.path.join(self.save_root, 'ap{}'.format(ap))
            torch.save(s_net.module.state_dict(), save_file_path)

            logger_file("val: Validation Results - Epoch: {} - LR: {}".format(epoch, optimizer.optimizer.param_groups[0]['lr']))
            print_summar_table(logger_file, attr_name, metrics_info['logger'])
            logger_file('AP:%0.3f' % metrics_info['logger']['attr']['ap'][-1])


parser = argparse.ArgumentParser(description='PyTorch my data Training')
args = parse_opts()
max_epoch = args.max_epoch - args.distill_epoch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log = Logger('both', filename=os.path.join(args.log_dir, args.log_file + '_all'), level='debug', mode='both')
logger = log.logger.info
log_config(args, logger)
log_file = Logger('file', filename=os.path.join(args.log_dir, args.log_file), level='debug', mode='file')
logger_file = log_file.logger.info
attr, attr_name = get_tasks(args)
criterion_CE, metrics = get_losses_metrics(attr, args.categorical_loss)

# Load dataset, net, evaluator, Saver
trainloader, testloader = get_data(args, attr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
t_net, s_net, channel_t, channel_s, layer_t, layer_s, index, filter_list = \
    get_pair_model(frm='my', name_t=args.name_t, name_s=args.name_s, pretrained_s=args.pretrained_s,
                   load_BN=args.load_BN, logger=logger, bucket=args.bucket, size=args.size)
if args.direct_connect:
    distill_net = AB_distill_Mobilenetl2MobilenetsNoConnect(t_net, s_net, args.batch_size, args.DTL,
        args.AB_loss_multiplier, args.DTL_loss_multiplier, channel_t, channel_s, layer_t, layer_s, criterion_CE, index, args.DTL_loss)
else:
    distill_net = AB_distill_Mobilenetl2Mobilenets(t_net, s_net, args.batch_size, args.DTL, args.AB_loss_multiplier,
        args.DTL_loss_multiplier, channel_t, channel_s, layer_t, layer_s, criterion_CE, args.stage1, args.DTL_loss)
if device == 'cuda':
    s_net = torch.nn.DataParallel(s_net).cuda()
    distill_net = torch.nn.DataParallel(distill_net).cuda()
    cudnn.benchmark = True
params_list = [{'params': filter(lambda p: id(p) not in filter_list, s_net.parameters())}]
if args.stage1 and not args.direct_connect:
    params_list.append({'params': distill_net.module.Connectors.parameters()})
optimizer = optim.SGD(params_list, lr=0.01, nesterov=True, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = MultiStepLR(optimizer, milestones=[5, 30], gamma=1)
train_evaluator = create_supervised_evaluator(s_net, metrics={
    'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)
val_evaluator = create_supervised_evaluator(s_net, metrics={
    'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)

Saver = Saver()
for epoch in range(1, int(args.distill_epoch) + 1):
    Distillation(distill_net, epoch)

# Cross-entropy training
distill_net.module.stage1 = False
base_params = filter(lambda p: id(p) not in filter_list, s_net.parameters())
spci_params = filter(lambda p: id(p) in filter_list, s_net.parameters())
params_list = [{'params': base_params}, {'params': spci_params, 'lr': args.lr*0.01}]
optimizer = optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
if args.scheduler == 'step':
    optimizer = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
elif args.scheduler == 'cos':
    optimizer = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-3)
elif args.scheduler == 'pleau':
    optimizer = ReduceLROnPlateau(optimizer, mode='max', patience=3)
else:
    raise Exception('not implement scheduler')

for epoch in range(1, max_epoch+1):
    if args.DTL:
        train_DTL(distill_net, epoch)
    else:
        train(s_net, epoch)
    metrics_info = test(s_net, epoch)
    Saver.save(epoch, metrics_info)
