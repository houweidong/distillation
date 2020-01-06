"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math
import torch
import os
from utils.get_channels import get_channels, get_name_of_alpha_and_beta

__all__ = ['get_model', 'get_pair_model', 'MobileNetV3']
root = os.environ['HOME']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1., num_attr=5, dropout=0.1):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.num_attr = num_attr
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = nn.Sequential(
            # conv_1x1_bn(input_channel, _make_divisible(exp_size * width_mult, 8)),
            conv_1x1_bn(input_channel, 960),
            # SELayer(960) if mode == 'small' else nn.Sequential()
            nn.Sequential()
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 512
        self.classifier = nn.ModuleList()
        for i in range(self.num_attr):
            classifier = nn.Sequential(
                # nn.Linear(_make_divisible(exp_size * width_mult, 8), output_channel),
                nn.Linear(960, output_channel),
                # nn.BatchNorm1d(output_channel) if mode == 'small' else nn.Sequential(),
                h_swish(),
                nn.Dropout(0.1),
                nn.Linear(output_channel, num_classes),
                # nn.BatchNorm1d(num_classes) if mode == 'small' else nn.Sequential(),
                # h_swish() if mode == 'small' else nn.Sequential()
            )
            self.classifier.append(classifier)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        result = []
        for i in range(self.num_attr):
            y = self.classifier[i](x)
            result.append(y)
        return torch.cat(result, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_channels_for_distill(cfgs):

    channels = []
    layers = []
    for i, cfg in enumerate(cfgs):
        if cfg[-1] == 2:
            channels.append(cfg[1])
            layers.append(i+1)
    return channels, layers


def mobile3l(**kwargs):
    frm = kwargs['frm'] if 'frm' in kwargs else 'my'
    device = kwargs['device'] if 'device' in kwargs else 'cuda'
    pretrained = kwargs['pretrained_t'] if 'pretrained_t' in kwargs else True
    name_t = kwargs['name_t'] if 'name_t' in kwargs else None
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 0, 0, 1],  # 1
        [3,  64,  24, 0, 0, 2],  # 2                    layer2  64
        [3,  72,  24, 0, 0, 1],  # 3
        [5,  72,  40, 1, 0, 2],  # 4                    layer4  72
        [5, 120,  40, 1, 0, 1],  # 5
        [5, 120,  40, 1, 0, 1],  # 6
        [3, 240,  80, 0, 1, 2],  # 7                    layer7  240
        [3, 200,  80, 0, 1, 1],  # 8
        [3, 184,  80, 0, 1, 1],  # 9
        [3, 184,  80, 0, 1, 1],  # 10
        [3, 480, 112, 1, 1, 1],  # 11
        [3, 672, 112, 1, 1, 1],  # 12
        [5, 672, 160, 1, 1, 1],  # 13
        [5, 672, 160, 1, 1, 2],  # 14                   layer14 672
        [5, 960, 160, 1, 1, 1]   # 15
    ]
    model = MobileNetV3(cfgs, mode='large')

    if pretrained:
        print('loading model from {}'.format(name_t))
        path = os.path.join(root, '.torch/models/', name_t)
        state_dict = torch.load(path, map_location=device)
        strict = False if frm == 'official' else True
        model.load_state_dict(state_dict, strict=strict)
        print('load completed')
    channels, layers = get_channels_for_distill(cfgs)
    return model, channels, layers


def mobile3s(**kwargs):
    device = kwargs['device'] if 'device' in kwargs else 'cuda'
    pretrained = kwargs['pretrained_s'] if 'pretrained_s' in kwargs else True
    name_s = kwargs['name_s'] if 'name_s' in kwargs else None
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 1, 0, 2],  # 1                    layer1  16
        [3,  72,  24, 0, 0, 2],  # 2                    layer2  72
        [3,  88,  24, 0, 0, 1],  # 3
        [5,  96,  40, 1, 1, 2],  # 4                    layer4  96
        [5, 240,  40, 1, 1, 1],  # 5
        [5, 240,  40, 1, 1, 1],  # 6
        [5, 120,  48, 1, 1, 1],  # 7
        [5, 144,  48, 1, 1, 1],  # 8
        [5, 288,  96, 1, 1, 2],  # 9                    layer9  288
        [5, 576,  96, 1, 1, 1],  # 10
        [5, 576,  96, 1, 1, 1],  # 11
    ]

    model = MobileNetV3(cfgs, mode='small')

    if pretrained:
        print('loading model from {}'.format(name_s))
        path = os.path.join(root, '.torch/models/', name_s)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print('load completed')
    channels, layers = get_channels_for_distill(cfgs)
    return model, channels, layers


def mobile3ss(**kwargs):
    # device = kwargs['device'] if 'device' in kwargs else 'cuda'
    # pretrained = kwargs['pretrained'] if 'pretrained' in kwargs else True
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 0, 0, 1],  # 1
        [3,  64,  24, 0, 0, 2],  # 2                    layer2  64
        [3,  72,  24, 0, 0, 1],  # 3
        [5,  72,  40, 1, 0, 2],  # 4                    layer4  72
        [5, 120,  40, 1, 0, 1],  # 5
        [3, 240,  80, 0, 1, 2],  # 6                    layer6  240
        [3, 200,  80, 0, 1, 1],  # 7
        [3, 480, 112, 1, 1, 1],  # 8
        [5, 672, 160, 1, 1, 1],  # 9
        [5, 672, 160, 1, 1, 2],  # 10                   layer10 672
        [5, 960, 160, 1, 1, 1]   # 11
    ]
    model = MobileNetV3(cfgs, mode='small')

    # if pretrained:
    #     path = os.path.join(root, '.torch/models/mobilenetv3-small-c7eb32fe.pth')
    #     state_dict = torch.load(path, map_location=device)
    #     model.load_state_dict(state_dict, strict=True)

    channels, layers = get_channels_for_distill(cfgs)
    return model, channels, layers


def get_model(conv, **kwargs):
    model = {'mobile3l': mobile3l, 'mobile3s': mobile3s, 'mobile3ss': mobile3ss}
    if conv not in model:
        raise Exception('not implemented model')
    return model[conv](**kwargs)


def get_pair_model(**kwargs):
    device = kwargs['device'] if 'device' in kwargs else 'cuda'
    name_t = kwargs['name_t']
    name_s = kwargs['name_s']
    mode = kwargs['mode']
    load_BN = kwargs['load_BN']

    cfgs_t = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 0, 0, 1],  # 1
        [3,  64,  24, 0, 0, 2],  # 2                    layer2  64
        [3,  72,  24, 0, 0, 1],  # 3
        [5,  72,  40, 1, 0, 2],  # 4                    layer4  72
        [5, 120,  40, 1, 0, 1],  # 5
        [5, 120,  40, 1, 0, 1],  # 6
        [3, 240,  80, 0, 1, 2],  # 7                    layer7  240
        [3, 200,  80, 0, 1, 1],  # 8
        [3, 184,  80, 0, 1, 1],  # 9
        [3, 184,  80, 0, 1, 1],  # 10
        [3, 480, 112, 1, 1, 1],  # 11
        [3, 672, 112, 1, 1, 1],  # 12
        [5, 672, 160, 1, 1, 1],  # 13
        [5, 672, 160, 1, 1, 2],  # 14                   layer14 672
        [5, 960, 160, 1, 1, 1]   # 15
    ]
    cfgs_s = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 1, 0, 2],  # 1                    layer1  16
        [3,  72,  24, 0, 0, 2],  # 2                    layer2  72
        [3,  88,  24, 0, 0, 1],  # 3
        [5,  96,  40, 1, 1, 2],  # 4                    layer4  96
        [5, 240,  40, 1, 1, 1],  # 5
        [5, 240,  40, 1, 1, 1],  # 6
        [5, 120,  48, 1, 1, 1],  # 7
        [5, 144,  48, 1, 1, 1],  # 8
        [5, 288,  96, 1, 1, 2],  # 9                    layer9  288
        [5, 576,  96, 1, 1, 1],  # 10
        [5, 576,  96, 1, 1, 1],  # 11
    ]
    model_t = MobileNetV3(cfgs_t, mode='large')
    channels_t, layers_t = get_channels_for_distill(cfgs_t)
    print('loading model from {}'.format(name_t))
    path_t = os.path.join(root, '.torch/models/', name_t)
    state_dict_t = torch.load(path_t, map_location=device)
    print('load completed')

    model_s = MobileNetV3(cfgs_s, mode='small')
    channels_s, layers_s = get_channels_for_distill(cfgs_s)
    index, alpha, beta = get_channels(state_dict_t, layers_t, channels_s, 'uniform')

    print('loading model from {}'.format(name_s))
    path_s = os.path.join(root, '.torch/models/', name_s)
    state_dict_s = torch.load(path_s, map_location=device)

    print('update last conv and classifier param in state_dict from teacher')
    for k in list(state_dict_s.keys()):
        # exclude the last conv's SE layer(for reducing params num) and
        # classifier layer(for performance and align problem)
        if k.startswith('classifier') or k.startswith('conv.1.fc.'):
            state_dict_s.pop(k)
    for k, v in state_dict_t.items():
        # load classifier parmas and the last conv's BN params
        if k.startswith('classifier') or k.startswith('conv.0.1.'):
            state_dict_s[k] = v
        # load the conv's conv params from teacher because the align problem
        state_dict_s['conv.0.0.weight'] = state_dict_t['conv.0.0.weight'][:, 0:cfgs_t[-1][2]//cfgs_s[-1][2]*cfgs_s[-1][2]:cfgs_t[-1][2]//cfgs_s[-1][2], :, :]
    print('update the last conv classifier param completed')

    if load_BN:
        print('update distill BN param in state_dict from teacher')
        for i in range(len(layers_s)):
            if i != 0:
                alpha_sn, beta_sn = get_name_of_alpha_and_beta(layers_s[i])
                state_dict_s[alpha_sn] = alpha[i]
                state_dict_s[beta_sn] = beta[i]
            else:
                alpha_sn, beta_sn = 'features.0.1.weight', 'features.0.1.bias'
                state_dict_s[alpha_sn] = alpha[i]
                state_dict_s[beta_sn] = beta[i]
        print('update distill BN param completed')
        model_s.load_state_dict(state_dict_s, strict=True)
    print('load student completed')

    # the last conv's bn layers has strong correlation with the features used to classify, so
    # this layer is divided to the classification's params
    last_conv_bn_ids = list(map(id, model_s.conv[0][1].parameters()))
    classifier_ids = list(map(id, model_s.classifier.parameters())) + last_conv_bn_ids

    BN_ids = []
    for i in range(len(layers_s)):
        if i != 0:
            BN_id = list(map(id, model_s.features[layers_s[i]].conv[1].parameters()))
        else:
            BN_id = list(map(id, model_s.features[0][1].parameters()))
        BN_ids.extend(BN_id)

    if load_BN:
        ids_list = classifier_ids + BN_ids
    else:
        ids_list = classifier_ids
    if mode == 'student':
        return model_s, classifier_ids
    else:
        model_t.load_state_dict(state_dict_t, strict=True)
        return model_t, model_s, channels_t, channels_s, layers_t, layers_s, index, ids_list
