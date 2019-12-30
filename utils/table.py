import collections
import math
from prettytable import PrettyTable


def print_summar_table(logger, attr_display_names, logger_info):
    title_list = collections.OrderedDict()
    title_list['attr'] = ['attr']
    if len(logger_info['rec']['ap']) != 0:
        title_list['rec'] = ['rec']

    for name in attr_display_names:
        if name.endswith('recognizable'):
            title_list['rec'].append(name.split('/')[0])
        else:
            title_list['attr'].append(name)
    for key in title_list:
        title_list[key].append('summary')

    for key in title_list:
        x = PrettyTable(title_list[key])
        x.align[key] = 'l'
        for key_info, item in logger_info[key].items():
            row = [key_info]
            row.extend(item)
            x.add_row(row)
        logger(x)


class TableForPrint(object):
    def __init__(self):
        self.logger_print, self.metrics, self.summary = self.create()

    # use 'None' placeholder if one recognition or attr have no data for a eval func
    def reset(self, name):
        if name.endswith('/recognizable'):
            for ap_acc_loss in self.logger_print['rec']:
                self.logger_print['rec'][ap_acc_loss].append('None')
        else:
            for ap_acc_loss in self.logger_print['attr']:
                self.logger_print['attr'][ap_acc_loss].append('None')

    def update(self, attr_detect, ap_acc_loss, val):
        if attr_detect.endswith('/recognizable'):
            self.logger_print['rec'][ap_acc_loss][-1] = float('{:.4f}'.format(val))
        else:
            self.logger_print['attr'][ap_acc_loss][-1] = float('{:.4f}'.format(val))
        metrics_name = attr_detect + '/' + ap_acc_loss
        self.metrics[metrics_name] = val

    def summarize(self):
        if len(self.logger_print['rec']['ap']) == 0:
            name_list = ['attr']
        else:
            name_list = ['attr', 'rec']
        fil = {}
        for name in name_list:
            for key, item in self.logger_print[name].items():
                fil[key] = list(filter(lambda x: x is not 'None' and not math.isnan(x), item))
            ap = float('{:.4f}'.format(sum(fil['ap']) / len(fil['ap']))) if len(fil['ap']) else float('nan')
            accuracy = float('{:.4f}'.format(sum(fil['accuracy']) / len(fil['accuracy']))) \
                if len(fil['accuracy']) else float('nan')
            # loss = float('{:.4f}'.format(sum(fil['loss'])))
            # for evl, evl_name in zip(['ap', 'accuracy', 'loss'], ['mAP', 'mAccuracy', 'total_loss']):
            for evl, evl_name in zip(['ap', 'accuracy'], ['mAP', 'mAccuracy']):
                self.logger_print[name][evl].append(locals()[evl])
                summary_name = name + '/' + evl_name
                # if name == 'rec':
                self.summary[summary_name] = locals()[evl]
                # if summary_name == 'rec/total_loss':
                #     self.summary['total_loss'] = self.summary['attr/total_loss'] + self.summary['rec/total_loss']

    def create(self):
        logger_for_print = collections.OrderedDict()
        # name_list = ['ap', 'accuracy', 'loss']
        name_list = ['ap', 'accuracy']
        for name in name_list:
            logger_for_print[name] = []

        logger_for_print_detect = collections.OrderedDict()
        for name in name_list:
            logger_for_print_detect[name] = []

        logger_print = collections.OrderedDict()
        logger_print['attr'] = logger_for_print
        logger_print['rec'] = logger_for_print_detect

        metrics = {}
        summary = {}
        return logger_print, metrics, summary
