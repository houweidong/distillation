import torch
from utils.get_tasks import get_tasks
from models.generate_model import get_model
from utils.opts import parse_opts


def jit_trace(args, model_path):
    attr, attr_name = get_tasks(args)
    model, _, _ = get_model(args.conv, classifier=args.classifier, dropout=args.dropout, attr=attr)
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    example = torch.rand(1, 3, 224, 224)
    a = torch.jit.trace(model.eval(), example)
    a.save('{}.pt'.format(args.conv))
    print('transform succeed')


if __name__ == '__main__':

    arg = parse_opts()
    model_path = '/root/.torch/models/modelofnewdata'
    jit_trace(arg, model_path)

