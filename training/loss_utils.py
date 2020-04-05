import torch
import torch.nn.functional as F
from functools import partial
from data.attributes import NewAttributes
from torch.nn import MSELoss, KLDivLoss
from torch.autograd import Variable


def exp_loss(pred, alpha=-23, beta=-18):
    loss = 0.0 * torch.exp(alpha * (pred + beta / float(7 * 7)))
    return loss.mean()


# alpha now only support for binary classification
# TODO Change it to class so that gamma can also be learned
def focal_loss(pred, target_float, gamma=2, alpha=None, size_average=True):
    target = torch.round(target_float).long()
    if isinstance(alpha, (float, int)):
        alpha = torch.Tensor([1 - alpha, alpha])
    if isinstance(alpha, list):
        alpha = torch.Tensor(alpha)

    target = target.view(-1, 1)
    target_float = target_float.view(-1, 1)
    # logpt = F.log_softmax(pred, 1)
    # logpt = logpt.gather(1, target)
    # pt = logpt.exp()
    # ls = F.logsigmoid(pred)
    # ls_1m = 1 - ls
    pt = torch.sigmoid(pred)
    pt_1m = 1 - pt

    logpt = torch.log(torch.cat((pt_1m, pt), dim=1)).gather(1, target)
    logpt = logpt.view(-1)

    pt_final = torch.cat((pt_1m, pt), dim=1).gather(1, target).view(-1)

    if alpha is not None:
        if alpha.type() != pred.data.type():
            alpha = alpha.type_as(pred.data)
        at = alpha.gather(0, target.data.view(-1))
        at = 1 - at
        logpt = logpt * at
    loss1_coe = torch.cat((1 - target_float, target_float), dim=1).gather(1, target)
    loss1 = (-1 * (1 - pt_final) ** gamma * logpt) * loss1_coe

    logpt1 = torch.log(torch.cat((pt, pt_1m), dim=1)).gather(1, target)
    logpt1 = logpt1.view(-1)
    pt_final = torch.cat((pt, pt_1m), dim=1).gather(1, target).view(-1)

    if alpha is not None:
        logpt1 = logpt1 * at
    loss2 = (-1 * (1 - pt_final) ** gamma * logpt1) * (1 - loss1_coe)
    # loss2_coe = 1 - loss1_coe
    loss = loss1 + loss2
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def binary_cn(pred, target, weight=None):
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)


# to solve the imbalance problem
def ohem_loss(pred, target_float, ratio=3, reverse=False):
    assert pred.size()[1] == 2 or pred.size()[1] == 1  # Only support binary case
    target = torch.round(target_float)
    # print(target)
    if not reverse:
        pos_mask = target.bool()
        neg_mask = ~pos_mask
    else:  # Simply reversing mask for positive/negative samples will yield reverse OHEM
        neg_mask = target.bool()
        pos_mask = ~neg_mask

    n_pos = int(torch.sum(pos_mask))
    n_neg = int(torch.sum(neg_mask))
    if n_neg > 0 and n_neg > n_pos * ratio:

        n_selected = max(n_pos * ratio, 1)

        ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target_float, reduction='none')
        # ce_loss = F.cross_entropy(pred, target, reduction='none')

        # generate top k neg ce loss mask
        loss_neg_samples = torch.masked_select(ce_loss, neg_mask)
        _, index = torch.topk(loss_neg_samples, n_selected)

        # Get mask of selected negative samples on original mask tensor
        selected_neg_mask = torch.zeros(int(n_neg)).bool().cuda()
        selected_neg_mask.scatter_(0, index, True)  # a [n_neg] size mask
        # print(n_pos, n_neg, neg_mask.size())
        neg_index = torch.masked_select(
            torch.arange(n_pos + n_neg, dtype=torch.long, device='cuda', requires_grad=False),
            neg_mask)  # Mapping from [n_neg] to [n_pos+n_neg] mask
        neg_mask.scatter_(0, neg_index, selected_neg_mask)
        # Return average loss of all selected samples
        mask = neg_mask | pos_mask
        masked_loss = torch.masked_select(ce_loss)

        # anp = torch.masked_select(pred[:, 0], neg_mask).mean()
        # app = torch.masked_select(pred[:, 1], pos_mask).mean()
        # np_contrast = anp / app
        return masked_loss.mean()  # , np_contrast
    else:
        # anp = torch.masked_select(pred[:, 0], neg_mask).mean()
        # app = torch.masked_select(pred[:, 1], pos_mask).mean()
        # np_contrast = anp / app
        # return F.cross_entropy(pred, target)  # , np_contrast
        return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())


def reverse_ohem_loss(pred, target, ratio=3): return ohem_loss(pred, target, ratio, reverse=True)


def get_categorial_loss(attrs, loss):
    if loss == 'cross_entropy':
        loss_fns = {}
        for attr in attrs:
            loss_fns[attr] = {}
            loss_fns[attr]['attr'] = binary_cn
            if attr.rec_trainable:
                loss_fns[attr]['rec'] = binary_cn
        return loss_fns
    elif loss == 'cross_entropy_weight':
        # return F.cross_entropy, F.cross_entropy
        loss_fns = {}
        weights = get_categorial_weight()
        for attr in attrs:
            loss_fns[attr] = {}
            loss_fns[attr]['attr'] = partial(binary_cn, weight=weights[attr.key][0])
            if attr.rec_trainable:
                loss_fns[attr]['rec'] = partial(binary_cn, weight=weights[attr.key][1])
        return loss_fns
    elif loss == 'ohem':
        loss_fns = {}
        for attr in attrs:
            loss_fns[attr] = {}
            loss_fns[attr]['attr'] = ohem_loss
            if attr.rec_trainable:
                loss_fns[attr]['rec'] = reverse_ohem_loss
        return loss_fns
        # return Ohem, ohem_loss
    elif loss == 'focal':
        loss_fns = {}
        weights = get_categorial_weight()
        for attr in attrs:
            loss_fns[attr] = {}
            loss_fns[attr]['attr'] = partial(focal_loss, alpha=weights[attr.key][0] / (weights[attr.key][0] + 1))
            if attr.rec_trainable:
                loss_fns[attr]['rec'] = partial(focal_loss, alpha=weights[attr.key][1] / (weights[attr.key][1] + 1))
        return loss_fns
        # return focal_loss, focal_loss
    else:
        raise Exception("Loss '{}' is not supported".format(loss))


# def get_categorial_scale():
#
#     scales = [(10263+2032)/16436, (19092+3243)/6396, (26284+991)/1456, (21674+422)/6635, (20991+1947)/5793,
#                 (13339+1879)/13513, (26200+273)/2258, (14120+10369)/4242, (18731+7585)/2415, (8168+10010)/10553,
#                 (18275+7571)/2885, (26622+1101)/1008, (19045+1252)/8434, (26507+229)/1995]
#
#     pos_num = [16436, 6396, 1456, 6635, 5793, 13513, 2258, 4242, 2415, 10553, 2885, 1008, 8434, 1995]
#     result = []
#     for scale in scales:
#         # result.append(1/(1+scale))
#         # if 0 <= scale < 5:
#         #     result.append(0.5)
#         # elif 5 <= scale < 10:
#         #     result.append(1/3)
#         # elif 10 <= scale:
#         #     result.append(0.25)
#         result.append(scale)
#
#     return result, pos_num


def get_categorial_weight():
    weight = {}
    weight[NewAttributes.yifujinshen_yesno] = [(26303) / 5303, 1064 / (5303 + 26303)]
    weight[NewAttributes.kuzijinshen_yesno] = [(13626) / 7255, 11789 / (7255 + 13626)]
    weight[NewAttributes.maozi_yesno] = [(20039) / 3494, 9137 / (3494 + 20039)]
    weight[NewAttributes.gaolingdangbozi_yesno] = [(21840) / 8501, 2329 / (8501 + 21840)]
    weight[NewAttributes.gaofaji_yesno] = [(11070) / 9039, 12561 / (11070 + 9039)]

    # just for classification for two class
    return weight


def get_weight():
    import math
    numbers_sample = [[1554, 28426], [2094, 6704], [2734, 14115], [3413, 21947],
                      [2248, 27422], [4735, 24261], [4664, 22594], [1783, 7511, 11486]]
    pos_ratio = torch.FloatTensor([0.264, 0.535, 0.175, 0.388, 0.815])
    w_p = (1 - pos_ratio).exp().cuda()
    w_n = pos_ratio.exp().cuda()
    floatTensorList = []
    for numbers in numbers_sample:
        if len(numbers) == 2:
            floatTensorList.append(
                torch.tensor([math.exp(numbers[1] / sum(numbers)), math.exp(numbers[0] / sum(numbers))],
                             dtype=torch.float, device='cuda', requires_grad=False))
        else:
            floatTensorList.append(torch.tensor([math.exp((1 - numbers[0] / sum(numbers)) * 3 / 4),
                                                 math.exp((1 - numbers[1] / sum(numbers)) * 3 / 4),
                                                 math.exp((1 - numbers[2] / sum(numbers)) * 3 / 4)],
                                                dtype=torch.float, device='cuda', requires_grad=False))
    return floatTensorList


# my masked loss for multi class
def multitask_loss(output, label, loss_fns):
    target, mask = label
    n_tasks_all = len(target)
    loss = 0
    for i in range(n_tasks_all):
        # Only add loss regarding this attribute if it is present in any sample of this batch
        if mask[i].any():
            output_fil = torch.masked_select(output[:, i],
                                             mask[i].squeeze(1).bool())  # .view(-1, output[i].size(1))
            gt = torch.masked_select(target[i].squeeze(1),
                                     mask[i].squeeze(1).bool())  # .view(-1, target[i].size(1))
            loss += loss_fns[i](output_fil, gt)
    return loss


def multitask_loss_balance(output, label, w_p, w_n):
    target, mask = label
    n_tasks_all = len(target)
    batch_size = target[0].size(0)
    loss = 0
    for i in range(n_tasks_all):
        # Only add loss regarding this attribute if it is present in any sample of this batch
        if mask[i].any():
            mask_temp = mask[i].repeat((4, 1))
            target_temp = target[i].repeat((4, 1))
            output_fil = torch.masked_select(output[:, i],
                                             mask_temp.squeeze(1).bool())  # .view(-1, output[i].size(1))
            gt = torch.masked_select(target_temp.squeeze(1),
                                     mask_temp.squeeze(1).bool())  # .view(-1, target[i].size(1))
            L = gt.size(0)
            w = torch.zeros(L).cuda()
            w[gt.data == 1] = w_p[i]
            w[gt.data == 0] = w_n[i]
            w = Variable(w, requires_grad=False)
            temp = - w * (gt * (1 / (1 + (-output_fil).exp())).log() + \
                          (1 - gt) * ((-output_fil).exp() / (1 + (-output_fil).exp())).log())
            loss += temp.sum()
    loss = loss / batch_size / n_tasks_all
    return loss


def multitask_loss_balance_new(output, label, w_list):
    # w_list: shape:[attr1, attr2, attr3]  attr1:[p, n] or [c1, c2, c3]
    target, mask = label
    n_tasks_all = len(target)
    batch_size = target[0].size(0)
    loss = 0
    for i in range(n_tasks_all):
        # Only add loss regarding this attribute if it is present in any sample of this batch
        if mask[i].any():
            mask_temp = mask[i].repeat((4, 1))
            target_temp = target[i].repeat((4, 1))
            output_fil = output[i].squeeze(1)[mask_temp.squeeze(1).bool()]
            gt = target_temp.squeeze(1)[mask_temp.squeeze(1).bool()]
            if i == n_tasks_all - 1:
                # for gaofaji
                temp = F.cross_entropy(output_fil, gt.long(), w_list[i], reduction='sum')
            else:
                L = gt.size(0)
                w = torch.zeros(L, dtype=torch.float, device='cuda', requires_grad=False).cuda()
                w[gt.data == 1] = w_list[i][0]
                w[gt.data == 0] = w_list[i][1]
                temp = F.binary_cross_entropy_with_logits(output_fil, gt, w, reduction='sum')
            loss += temp
    loss = loss / batch_size / n_tasks_all
    return loss

# def SigmoidCrossEntropyLoss(x, y, w_p, w_n):
# 	# weighted sigmoid cross entropy loss defined in Li et al. ACPR'15
# 	loss = 0.0
# 	if not x.size() == y.size():
# 		print("x and y must have the same size")
# 	else:
# 		N = y.size(0)
# 		L = y.size(1)
# 		for i in range(N):
# 			w = torch.zeros(L).cuda()
# 			w[y[i].data == 1] = w_p[y[i].data == 1]
# 			w[y[i].data == 0] = w_n[y[i].data == 0]
#
# 			w = Variable(w, requires_grad = False)
# 			temp = - w * ( y[i] * (1 / (1 + (-x[i]).exp())).log() + \
# 				(1 - y[i]) * ( (-x[i]).exp() / (1 + (-x[i]).exp()) ).log() )
# 			loss += temp.sum()
#
# 		loss = loss / N
# 	return loss
