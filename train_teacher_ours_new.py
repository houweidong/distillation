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
from training.loss_utils import multitask_loss, multitask_loss_balance_new
from ignite.engine import create_supervised_evaluator
from training.metric_utils import MultiAttributeMetricNew
from training.get_loss_metric import get_losses_metrics
from utils.my_engine import my_evaluator
from utils.table import print_summar_table
from utils.logger import Logger
from utils.opts import parse_opts
from utils.log_config import log_config
from training.loss_utils import get_weight

# Training
def train(net, epoch, criterion, w_list, all_two=False):
    # epoch_start_time = time.time()
    logger('\nClassification training Epoch: %d' % epoch)
    net.train()
    # train_loss = 0
    bt_sum = len(trainloader)
    logger('lr: %.4f' % optimizer.optimizer.param_groups[0]['lr'])
    for batch_idx, bt in enumerate(trainloader):
        data_length = len(trainloader)
        inputs, targets = _prepare_batch(bt, device=device) if device == 'cuda' else bt
        batch_size = inputs[0].size(0)
        inputs_l = torch.cat(inputs[:2])
        inputs_s = torch.cat(inputs[2:])

        outputs_l = net(inputs_l)
        outputs_s = net(inputs_s)

        outputs = []
        for output_l, output_s in zip(outputs_l, outputs_s):
            outputs.append(torch.cat((output_l, output_s)))
        loss = criterion(outputs, targets, w_list, all_two)
        output_list = []
        for op in outputs:
            output_list.append(op.split(batch_size))

        losses = 0
        losses += loss
        logger_str_ot = ''
        losses_ot_items = []
        for i in range(4):
            for j in range(i + 1, 4):
                loss_ot = 0
                for k in range(len(output_list)):
                    loss_ot += F.mse_loss(output_list[k][i], output_list[k][j])
                losses += loss_ot
                losses_ot_items.append(loss_ot.item())
                logger_str_ot += 'loss_ot{:>02d}{:>02d}'.format(i + 1, j + 1) + ':{:>6.3f}\t'
        optimizer.optimizer.zero_grad()
        losses.backward()
        optimizer.optimizer.step()
        # train_loss += loss.item()
        # logger('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
        if batch_idx % 20 == 0:
            logger(('epoch:{},\ttrain step:{:>4}/{}\tLoss:{:>6.3f}\tcls loss:{:>6.3f}\t\t' + logger_str_ot)
                   .format(epoch, batch_idx, data_length, losses.item(), loss.item(), *losses_ot_items)
                   )


def test(net, epoch):
    net.eval()
    data_list = [trainloader, testloader]
    name_list = ['train', 'val']
    eval_list = [train_evaluator, val_evaluator]

    for data, name, evl in zip(data_list, name_list, eval_list):
        # images, label = data[0], data[1]
        # data = [images[0], label]
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
            torch.save(t_net.module.state_dict(), save_file_path)

            logger_file("val: Validation Results - Epoch: {} - LR: {}".format(epoch, optimizer.optimizer.param_groups[0]['lr']))
            print_summar_table(logger_file, attr_name, metrics_info['logger'])
            logger_file('AP:%0.3f' % metrics_info['logger']['attr']['ap'][-1])


parser = argparse.ArgumentParser(description='PyTorch my data Training')
args = parse_opts()
log = Logger('both', filename=os.path.join(args.log_dir, args.log_file + '_all'), level='debug', mode='both')
logger = log.logger.info
log_config(args, logger, single=True)
log_file = Logger('file', filename=os.path.join(args.log_dir, args.log_file), level='debug', mode='file')
logger_file = log_file.logger.info
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model, dataloader, opmiter, use ignite's evaluator
attr, attr_name = get_tasks(args)
t_net, ids, _ = get_model(args.conv, frm='official', name_s=args.name_s, name_t=args.name_t, logger=logger,
                          pretrained_t=args.pretrained_t, pretrained_s=args.pretrained_s, device=device,
                          plug_in=args.plug_in, classifier=args.classifier, dropout=args.dropout, fc1=args.fc1, fc2=args.fc2, attr=attr)
if device == 'cuda':
    t_net = torch.nn.DataParallel(t_net).cuda()

_, metrics = get_losses_metrics(attr, args.categorical_loss)
trainloader, testloader = get_data(args, attr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

head_params = filter(lambda p: id(p) not in ids, t_net.parameters()) if args.freeze_backbone else t_net.parameters()
optimizer = optim.SGD(head_params, lr=args.lr, nesterov=args.nesterov, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.adam
if args.scheduler == 'step':
    optimizer = MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)
elif args.scheduler == 'cos':
    optimizer = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-3)
elif args.scheduler == 'pleau':
    optimizer = ReduceLROnPlateau(optimizer, mode='max', patience=3)
else:
    raise Exception('not implement scheduler')

train_evaluator = my_evaluator(t_net, metrics={
    'multitask': MultiAttributeMetricNew(metrics, attr_name, attr=attr)}, device=device)
val_evaluator = my_evaluator(t_net, metrics={
    'multitask': MultiAttributeMetricNew(metrics, attr_name, attr=attr)}, device=device)
Saver = Saver()

w_list = get_weight(all_two=args.all_two)
criterion = multitask_loss_balance_new

for epoch in range(1, args.n_epochs+1):
    train(t_net, epoch, criterion, w_list, args.all_two)
    metric_info = test(t_net, epoch)
    Saver.save(epoch, metric_info)
