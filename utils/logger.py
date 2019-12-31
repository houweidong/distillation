import logging
# from logging import handlers


class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    fmt = '%(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式

    def __init__(self, name, filename=None, level='info', mode='both'):
        # fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # filename there just a name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别

        handlers = []
        if mode in ['both', 'screen']:
            sh = logging.StreamHandler()  # 往屏幕上输出
            sh.setFormatter(self.format_str)  # 设置屏幕上显示的格式
            handlers.append(sh)
        if mode in ['both', 'file']:
            fh = logging.FileHandler(filename)
            fh.setFormatter(self.format_str)
            handlers.append(fh)
        for h in handlers:
            self.logger.addHandler(h)  # 把对象加到logger里

# if __name__ == '__main__':
#     log = Logger('ap_results.log', level='debug')
#     log.logger.debug('debug')
#     log.logger.info('info')
