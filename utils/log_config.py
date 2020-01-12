from utils.logger import Logger
from collections import OrderedDict
import os


def log_config(args, logger=None, single=False):
    if not logger:
        log = Logger('both', filename=os.path.join(args.log_dir, args.log_file), level='debug', mode='both')
        logger = log.logger.info

    log_names = ['conv_t', 'conv_s', 'pretrained_t', 'pretrained_s', 'name_t', 'name_s', 'DTL', 'stage1', 'load_BN',
                 'direct_connect', 'AB_loss_multiplier', 'DTL_loss_multiplier', 'distill_epoch', 'max_epoch',
                 'lr', 'nesterov', 'scheduler', 'bucket', 'size', 'freeze_backbone', 'plug_in', 'classifier', 'dropout']
    pop_list = []
    if single:
        pop_list = ['distill_epoch', 'max_epoch', 'conv_t', 'conv_s', 'DTL', 'stage1', 'direct_connect',
                    'AB_loss_multiplier', 'DTL_loss_multiplier', 'load_BN', 'bucket', 'size', 'freeze_backbone']
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
        logger('{:<30}{}'.format(name.upper(), args.__dict__[name]))
