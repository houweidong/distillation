import torch
import numpy as np


# [4, 5, 3, 0, 6, 13, 15, 8, 8, 2] [-0.56730014, -0.40980357, -0.252307, -0.09481044, 0.06268612, 0.22018269, 0.37767926, 0.5351758, 0.6926724, 0.85016894, 1.0076655]
# [2, 0, 4, 3, 9, 7, 11, 11, 10, 7] [-0.5200588, -0.38907784, -0.25809687, -0.1271159, 0.0038650632, 0.13484603, 0.265827, 0.39680797, 0.52778894, 0.6587699, 0.7897509]
# [-0.56730014 -0.5200588 ] [1.0076655 0.7897509]
#
# [1, 0, 0, 2, 4, 12, 24, 22, 4, 3] [-0.45488524, -0.31360936, -0.17233345, -0.03105756, 0.11021833, 0.25149423, 0.3927701, 0.534046, 0.67532194, 0.8165978, 0.9578737]
# [1, 0, 1, 1, 12, 17, 10, 8, 11, 11] [-1.0421665, -0.8872676, -0.7323687, -0.57746977, -0.42257088, -0.267672, -0.112773106, 0.042125784, 0.19702467, 0.35192358, 0.50682247]
# [-0.45488524 -1.0421665 ] [0.9578737  0.50682247]
#
# [1, 0, 15, 14, 16, 29, 71, 60, 26, 8] [-0.25704733, -0.12531526, 0.0064167976, 0.13814886, 0.26988092, 0.401613, 0.53334504, 0.6650771, 0.7968092, 0.92854124, 1.0602733]
# [6, 15, 36, 37, 25, 30, 15, 23, 42, 11] [-1.4335799, -1.2705797, -1.1075795, -0.9445793, -0.78157914, -0.6185789, -0.4555787, -0.29257852, -0.12957832, 0.03342187, 0.19642207]
# [-0.25704733 -1.4335799 ] [1.0602733  0.19642207]
#
# [19, 7, 18, 11, 15, 98, 225, 198, 70, 11] [0.05439103, 0.1345677, 0.21474436, 0.294921, 0.3750977, 0.45527434, 0.535451, 0.61562765, 0.69580436, 0.775981, 0.85615766]
# [5, 45, 178, 205, 80, 42, 27, 20, 18, 52] [-1.2577643, -1.1277652, -0.997766, -0.8677669, -0.73776776, -0.6077686, -0.47776946, -0.3477703, -0.21777117, -0.08777203, 0.042227123]
# [ 0.05439103 -1.2577643 ] [0.85615766 0.04222712]


def distance(a, max_pair, mode):
    """
    # type: (numpy, numpy, str) -> numpy
    compute distance between a and max_pair according to different mode

    :param a: shape is (n, 2)
    :param max_pair: [alpha, beta], shape is (2, )
    :param mode: high-mean: compute beta distance
                 high-var: compute alpha distance
                 high-all: compute both alpha and beta distance
    :return: distances, shape (N, )
    """
    if mode == 'high-mean':
        return np.power(a[:, 1] - max_pair[1], 2)
    elif mode == 'high-var':
        return np.power(a[:, 0] - max_pair[0], 2)
    elif mode == 'high-all':
        return np.power(a[:, 1] - max_pair[1], 2) + np.power(a[:, 0] - max_pair[0], 2)
    else:
        raise Exception('mode not supported')


def select_channel_ac_mean_var(alpha, beta, channel_nums, mode):
    """
    # type: (Tensor, Tensor, int, str) -> list
    Select some channels from all channels according to alpha and beta

    :param alpha: alpha of BN layer, alpha is tensor, shape is (N, )
    :param beta: beta of BN layer, beta is tensor, shape is (N, )
    :param channel_nums: the numbers of your selected channels
    :param mode: mode is a string type
                uniform: mean to uniformly select the channel
                high-mean: mean select the channels has higher beta but lower alpha
                high-var: mean select the channels has higher alpha but lower beta
                high-all: mean select the channels has higher beta and alpha
    :return index: index of selected channels, whose length is channel_nums
    """
    assert mode in ['uniform', 'high-mean', 'high-var', 'high-all']
    alpha = alpha.cpu().numpy()[:, np.newaxis]
    beta = beta.cpu().numpy()[:, np.newaxis]
    # alpha_bin = np.histogram(alpha, 10)
    # beta_bin = np.histogram(beta, 10)
    # max_pair = np.array([np.max(alpha), np.max(beta)])
    # min_pair = np.array([np.min(alpha), np.min(beta)])
    # if min_pair[0] >= 0. or min_pair[1] >= 0.:
    #     raise Exception('min larger than 0, the code should be updated')
    if mode != 'uniform':
        # TODO
        raise Exception('not implemented')
        pass
        # dist_pos = distance(np.concatenate((alpha, beta), axis=1), max_pair, mode)
        # dist_neg = distance(np.concatenate((alpha, beta), axis=1), min_pair, mode)
        # index = np.argsort(dist_pos)[-channel_nums:]
    else:
        length = len(alpha)
        step = length // channel_nums
        index = np.arange(0, step*channel_nums, step=step)
    return list(index)


def get_name_of_alpha_and_beta(layer):
    name_alpha = 'features.' + str(layer) + '.conv.' + str(1) + '.weight'
    name_beta = 'features.' + str(layer) + '.conv.' + str(1) + '.bias'
    return name_alpha, name_beta


def get_channels(model_dict, layers_t, channels_s, mode):
    indexs, alphas, betas = [], [], []
    for i, ly in enumerate(layers_t):
        alpha_n, beta_n = get_name_of_alpha_and_beta(layers_t[i])
        alpha, beta = model_dict[alpha_n], model_dict[beta_n]
        ind = select_channel_ac_mean_var(alpha, beta, channels_s[i], mode)
        alphas.append(alpha[ind])
        betas.append(beta[ind])
        indexs.append(ind)
    return indexs, alphas, betas


# path = '/root/.torch/models/ap0.8972'
# model = torch.load(path)
# cfgs_s = [
#     # k, t, c, SE, NL, s
#     [3, 16, 16, 1, 0, 2],  # 1                    layer1  16
#     [3, 72, 24, 0, 0, 2],  # 2                    layer2  72
#     [3, 88, 24, 0, 0, 1],  # 3
#     [5, 96, 40, 1, 1, 2],  # 4                    layer4  96
#     [5, 240, 40, 1, 1, 1],  # 5
#     [5, 240, 40, 1, 1, 1],  # 6
#     [5, 120, 48, 1, 1, 1],  # 7
#     [5, 144, 48, 1, 1, 1],  # 8
#     [5, 288, 96, 1, 1, 2],  # 9                    layer9  288
#     [5, 576, 96, 1, 1, 1],  # 10
#     [5, 576, 96, 1, 1, 1],  # 11
# ]
# cfgs_t = [
#     # k, t, c, SE, NL, s
#     [3, 16, 16, 0, 0, 1],  # 1
#     [3, 64, 24, 0, 0, 2],  # 2                    layer2  64
#     [3, 72, 24, 0, 0, 1],  # 3
#     [5, 72, 40, 1, 0, 2],  # 4                    layer4  72
#     [5, 120, 40, 1, 0, 1],  # 5
#     [5, 120, 40, 1, 0, 1],  # 6
#     [3, 240, 80, 0, 1, 2],  # 7                    layer7  240
#     [3, 200, 80, 0, 1, 1],  # 8
#     [3, 184, 80, 0, 1, 1],  # 9
#     [3, 184, 80, 0, 1, 1],  # 10
#     [3, 480, 112, 1, 1, 1],  # 11
#     [3, 672, 112, 1, 1, 1],  # 12
#     [5, 672, 160, 1, 1, 1],  # 13
#     [5, 672, 160, 1, 1, 2],  # 14                   layer14 672
#     [5, 960, 160, 1, 1, 1]  # 15
# ]
# cs, ls = get_channels_for_distill(cfgs_s)
# ct, lt = get_channels_for_distill(cfgs_t)
# ind, al, beta = get_channels(model, lt, cs)