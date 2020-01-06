import torch
from utils.get_tasks import get_tasks
from models.mobilenetv3 import get_model
import argparse


def jit_trace(conv, model_path):
    model, _, _ = get_model(conv)
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    example = torch.rand(1, 3, 224, 224)
    a = torch.jit.trace(model.eval(), example)
    a.save('{}.pt'.format(conv))
    print('transform succeed')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--conv', type=str, default='mobile3l', choices=['mobile3l', 'mobile3s', 'mobile3ss'])
    parse.add_argument('--model_path', type=str, default='')
    arg = parse.parse_args()
    jit_trace(arg.conv, arg.model_path)

