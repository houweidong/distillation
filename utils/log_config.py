from utils.logger import Logger
from collections import OrderedDict
import os


def log_config(args, logger=None, single=False):
    if not logger:
        log = Logger('both', filename=os.path.join(args.log_dir, args.log_file), level='debug', mode='both')
        logger = log.logger.info

    log_names = ['conv_t', 'conv_s', 'pretrained_t', 'pretrained_s', 'name_t', 'name_s', 'DTL', 'distill_epoch', 'max_epoch',
                 'lr', 'nesterov', 'scheduler']
    pop_list = []
    if single:
        pop_list = ['distill_epoch', 'max_epoch', 'conv_t', 'conv_s', 'DTL']
        if args.conv in ['mobile3l']:
            pop_list.extend(['pretrained_s', 'name_s'])
        elif args.conv in ['mobile3s', 'mobile3ss']:
            pop_list.extend(['pretrained_t', 'name_t'])
        else:
            raise Exception('args.conv error')
        log_names.extend(['weight_decay', 'n_epochs'])
    for name in pop_list:
        log_names.remove(name)
    for name in log_names:
        logger('{:<15}{}'.format(name.upper(), args.__dict__[name]))