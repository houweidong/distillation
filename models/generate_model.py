from models.mobilenetv3 import *
from models.faster_base import lsnet


def get_pair_model(size, **kwargs):
    model = {'s': get_pair_model_s, 'ss': get_pair_model_ss}
    return model[size](**kwargs)


def get_model(conv, **kwargs):
    model = {'mobile3l': mobile3l, 'mobile3s': mobile3s, 'mobile3ss': mobile3ss, 'lsnet': lsnet}
    if conv not in model:
        raise Exception('not implemented model')
    if conv != 'lsnet':
        return model[conv](**kwargs)
    else:
        model, ids = model[conv](**kwargs)
        return model, ids, None

