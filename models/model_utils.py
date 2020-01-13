import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_tanh(nn.Module):
    def __init__(self, inplace=True):
        super(h_tanh, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6 - 0.5


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


class ATLayer(nn.Module):
    def __init__(self, resolution, reduction=4):
        super(ATLayer, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(resolution*resolution, resolution*resolution // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(resolution*resolution // reduction, resolution*resolution),
                h_tanh()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = torch.mean(x, dim=1).view(b, w*h)
        y = self.fc1(y).view(b, 1, w, h)
        return x * (1 + y)


class ATSELayer(nn.Module):
    def __init__(self, channel, resolution, reduction=4):
        super(ATSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

        self.fc1 = nn.Sequential(
                nn.Linear(resolution*resolution, resolution*resolution // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(resolution*resolution // reduction, resolution*resolution),
                h_tanh()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        z = torch.mean(x, dim=1).view(b, w*h)
        z = self.fc1(z).view(b, 1, w, h)
        return x * y * (1 + z)


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


class Classifier(nn.Module):
    def __init__(self, num_attr, in_features, dropout=0.1, k=10, reduction=4):
        super(Classifier, self).__init__()
        self.num_attr = num_attr
        self.classifier = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        for i in range(self.num_attr):
            classifier = nn.Sequential(
                # nn.Linear(_make_divisible(exp_size * width_mult, 8), output_channel),
                nn.Linear(in_features, in_features),
                # nn.BatchNorm1d(output_channel) if mode == 'small' else nn.Sequential(),
                h_swish(),
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
                # nn.BatchNorm1d(num_classes) if mode == 'small' else nn.Sequential(),
                # h_swish() if mode == 'small' else nn.Sequential()
            )
            self.classifier.append(classifier)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        result = []
        for i in range(self.num_attr):
            y = self.classifier[i](x)
            result.append(y)
        result = torch.cat(result, dim=1)
        if not self.training:
            result = self.sigmoid(result)
        return result


class PrTp(nn.Module):
    def __init__(self, num_attr, in_features, dropout=0.1, k=10, reduction=4):
        super(PrTp, self).__init__()
        self.reduction = reduction
        self.in_features = in_features
        self.num_attr = num_attr
        self.relu = nn.ReLU6(inplace=True)
        self.h_sigmoid = h_sigmoid()
        self.h_tanh = h_tanh()
        self.activation = h_swish()
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        self.sigmoid = nn.Sigmoid()
        self.pt_avg = nn.AdaptiveAvgPool2d((1, 1))

        for i in range(self.num_attr):
            setattr(self, 'fc' + str(i) + '_1', nn.Linear(self.in_features, self.in_features))
            setattr(self, 'fc' + str(i) + '_2', nn.Linear(self.in_features, 1))
            # 10 prototype just for test
            setattr(self, 'attention' + str(i) + '_cv1', nn.Conv2d(self.in_features, self.in_features // self.reduction, (3, 3), padding=1))
            setattr(self, 'attention' + str(i) + '_cv1_bn1', nn.BatchNorm2d(self.in_features // self.reduction))
            setattr(self, 'attention' + str(i) + '_cv2', nn.Conv2d(self.in_features // self.reduction, k, (3, 3), padding=1))
            setattr(self, 'attention' + str(i) + '_cv2_bn2', nn.BatchNorm2d(k))

            setattr(self, 'prototype' + str(i) + '_coe1', nn.Linear(self.in_features, self.in_features // self.reduction))
            setattr(self, 'prototype' + str(i) + '_coe2', nn.Linear(self.in_features // self.reduction, k))

    def forward(self, x):
        results = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)
        for i in range(self.num_attr):
            cv1_rl = getattr(self, 'attention' + str(i) + '_cv1')(x)
            cv1_bn1 = self.activation(getattr(self, 'attention' + str(i) + '_cv1_bn1')(cv1_rl))
            cv2_rl = getattr(self, 'attention' + str(i) + '_cv2')(cv1_bn1)
            cv2_bn2 = getattr(self, 'attention' + str(i) + '_cv2_bn2')(cv2_rl)

            prototype_coe1 = self.relu(getattr(self, 'prototype' + str(i) + '_coe1')(self.pt_avg(x).view(batch, -1)))
            prototype_coe2 = self.h_tanh(getattr(self, 'prototype' + str(i) + '_coe2')(prototype_coe1))

            # multi prototype with attention map to produce new attention map
            new_attention = self.h_sigmoid((prototype_coe2[..., None, None] * cv2_bn2).sum(1, keepdim=True))
            y = new_attention * x
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = self.dropout(self.activation(getattr(self, 'fc' + str(i) + '_1')(y)))
            cls = getattr(self, 'fc' + str(i) + '_2')(y)
            results.append(cls)
        results = torch.cat(results, dim=1)
        if not self.training:
            results = self.sigmoid(results)
        return results


class CPrTp(nn.Module):
    def __init__(self, num_attr, in_features, dropout=0.1, k=10, reduction=4):
        super(CPrTp, self).__init__()
        self.reduction = reduction
        self.in_features = in_features
        self.num_attr = num_attr
        self.relu = nn.ReLU6(inplace=True)
        self.h_sigmoid = h_sigmoid()
        self.h_tanh = h_tanh()
        self.activation = h_swish()
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        self.sigmoid = nn.Sigmoid()
        self.pt_avg = nn.AdaptiveAvgPool2d((1, 1))

        setattr(self, 'attention' + '_cv1',
                nn.Conv2d(self.in_features, self.in_features // self.reduction, (3, 3), padding=1))
        setattr(self, 'attention' + '_cv1_bn1', nn.BatchNorm2d(self.in_features // self.reduction))
        setattr(self, 'attention' + '_cv2', nn.Conv2d(self.in_features // self.reduction, k, (3, 3), padding=1))
        setattr(self, 'attention' + '_cv2_bn2', nn.BatchNorm2d(k))

        for i in range(self.num_attr):
            setattr(self, 'fc' + str(i) + '_1', nn.Linear(self.in_features, self.in_features))
            setattr(self, 'fc' + str(i) + '_2', nn.Linear(self.in_features, 1))

            setattr(self, 'prototype' + str(i) + '_coe1',
                    nn.Linear(self.in_features, self.in_features // self.reduction))
            setattr(self, 'prototype' + str(i) + '_coe2', nn.Linear(self.in_features // self.reduction, k))

    def forward(self, x):
        results = []
        results_at = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)

        cv1_rl = getattr(self, 'attention' + '_cv1')(x)
        cv1_bn1 = self.activation(getattr(self, 'attention' + '_cv1_bn1')(cv1_rl))
        cv2_rl = getattr(self, 'attention' + '_cv2')(cv1_bn1)
        cv2_bn2 = getattr(self, 'attention' + '_cv2_bn2')(cv2_rl)

        for i in range(self.num_attr):
            prototype_coe1 = self.relu(getattr(self, 'prototype' + str(i) + '_coe1')(self.pt_avg(x).view(batch, -1)))
            prototype_coe2 = self.h_tanh(getattr(self, 'prototype' + str(i) + '_coe2')(prototype_coe1))

            # multi prototype with attention map to produce new attention map
            new_attention = self.h_sigmoid((prototype_coe2[..., None, None] * cv2_bn2).sum(1, keepdim=True))
            y = new_attention * x
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = self.dropout(self.activation(getattr(self, 'fc' + str(i) + '_1')(y)))
            cls = getattr(self, 'fc' + str(i) + '_2')(y)
            results.append(cls)
        results = torch.cat(results, dim=1)
        if not self.training:
            results = self.sigmoid(results)
        return results


class PCPrTp(nn.Module):
    def __init__(self, num_attr, in_features, dropout=0.1, k=10, reduction=4):
        super(PCPrTp, self).__init__()
        self.reduction = reduction
        self.in_features = in_features
        self.num_attr = num_attr
        self.relu = nn.ReLU6(inplace=True)
        self.h_sigmoid = h_sigmoid()
        self.h_tanh = h_tanh()
        self.activation = h_swish()
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        self.sigmoid = nn.Sigmoid()
        self.pt_avg = nn.AdaptiveAvgPool2d((1, 1))

        setattr(self, 'attention' + '_cv1',
                nn.Conv2d(self.in_features, self.in_features // self.reduction, (3, 3), padding=1))
        setattr(self, 'attention' + '_cv1_bn1', nn.BatchNorm2d(self.in_features // self.reduction))
        setattr(self, 'attention' + '_cv2', nn.Conv2d(self.in_features // self.reduction, k//2, (3, 3), padding=1))
        setattr(self, 'attention' + '_cv2_bn2', nn.BatchNorm2d(k//2))

        for i in range(self.num_attr):
            setattr(self, 'fc' + str(i) + '_1', nn.Linear(self.in_features, self.in_features))
            setattr(self, 'fc' + str(i) + '_2', nn.Linear(self.in_features, 1))
            # 10 prototype just for test
            setattr(self, 'attention' + str(i) + '_cv1',
                    nn.Conv2d(self.in_features, self.in_features // self.reduction, (3, 3), padding=1))
            setattr(self, 'attention' + str(i) + '_cv1_bn1', nn.BatchNorm2d(self.in_features // self.reduction))
            setattr(self, 'attention' + str(i) + '_cv2', nn.Conv2d(self.in_features // self.reduction, k//2, (3, 3), padding=1))
            setattr(self, 'attention' + str(i) + '_cv2_bn2', nn.BatchNorm2d(k//2))

            setattr(self, 'prototype' + str(i) + '_coe1',
                    nn.Linear(self.in_features, self.in_features // self.reduction))
            setattr(self, 'prototype' + str(i) + '_coe2', nn.Linear(self.in_features // self.reduction, k))

    def forward(self, x):
        results = []
        results_at = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)

        cv1_rl_cm = getattr(self, 'attention' + '_cv1')(x)
        cv1_bn1_cm = self.activation(getattr(self, 'attention' + '_cv1_bn1')(cv1_rl_cm))
        cv2_rl_cm = getattr(self, 'attention' + '_cv2')(cv1_bn1_cm)
        cv2_bn2_cm = getattr(self, 'attention' + '_cv2_bn2')(cv2_rl_cm)

        for i in range(self.num_attr):
            cv1_rl = getattr(self, 'attention' + str(i) + '_cv1')(x)
            cv1_bn1 = self.activation(getattr(self, 'attention' + str(i) + '_cv1_bn1')(cv1_rl))
            cv2_rl = getattr(self, 'attention' + str(i) + '_cv2')(cv1_bn1)
            cv2_bn2 = getattr(self, 'attention' + str(i) + '_cv2_bn2')(cv2_rl)

            prototype_coe1 = self.relu(getattr(self, 'prototype' + str(i) + '_coe1')(self.pt_avg(x).view(batch, -1)))
            prototype_coe2 = self.h_tanh(getattr(self, 'prototype' + str(i) + '_coe2')(prototype_coe1))

            cv2_bn2 = torch.cat([cv2_bn2_cm, cv2_bn2], 1)

            # multi prototype with attention map to produce new attention map
            new_attention = self.h_sigmoid((prototype_coe2[..., None, None] * cv2_bn2).sum(1, keepdim=True))
            y = new_attention * x
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = self.dropout(self.activation(getattr(self, 'fc' + str(i) + '_1')(y)))
            cls = getattr(self, 'fc' + str(i) + '_2')(y)
            results.append(cls)
        results = torch.cat(results, dim=1)
        if not self.training:
            results = self.sigmoid(results)
        return results
