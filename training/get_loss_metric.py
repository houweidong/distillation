from ignite.metrics import Loss
# from ignite.contrib.metrics import AveragePrecision
from training.loss_utils import get_categorial_loss
from data.attributes import AttributeType
# import torch.nn.functional as F
import torch
from functools import partial
from training.metric_utils import MyAccuracy, MyAveragePrecision


def get_losses_metrics(attrs, categorical_loss='cross_entropy'):

    # scales, pos_nums = get_categorial_scale()
    loss_fns = get_categorial_loss(attrs, categorical_loss)
    losses, metrics = [], []
    # cam_losses = []
    for attr in attrs:
        if attr.data_type == AttributeType.BINARY:
            # metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(loss_fn)])
            metrics.append(
                [MyAveragePrecision(output_transform=lambda pred, y: (pred, torch.round(y).long())),
                 MyAccuracy(output_transform=lambda pred, y: (pred, torch.round(y).long()))])
            losses.append(loss_fns[attr]['attr'])

        elif attr.data_type == AttributeType.MULTICLASS:
            for i in range(attr.branch_num):
                metrics.append(
                    [MyAveragePrecision(activation=lambda pred: torch.sigmoid(pred)),
                     MyAccuracy(output_transform=lambda pred: torch.sigmoid(pred))])
                losses.append(loss_fns[attr]['attr'])
        elif attr.data_type == AttributeType.NUMERICAL:
            # not support now
            pass
        # For recognizability classification
        if attr.rec_trainable:
            # metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(reverse_ohem_loss)])
            metrics.append(
                [MyAveragePrecision(activation=lambda pred, y: (torch.sigmoid(pred), torch.round(y).long())),
                 MyAccuracy(activation=lambda pred, y: (pred, torch.round(y).long()))])
            # Always use reverse OHEM loss for recognizability, at least for now
            losses.append(loss_fns[attr]['rec'])

    return losses, metrics
