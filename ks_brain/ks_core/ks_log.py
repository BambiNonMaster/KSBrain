import logging
from logging.handlers import TimedRotatingFileHandler
from .ks_utility import singleton

"""
author: mengru.du
create: 2022.3.12
modify: 20220312

该文件定义了日志类
1. 初始化日志对象
2. 默认添加控制台输出功能
3. 其他程序调用时通过 add_file_log 方法增加日志保存本地功能
"""


@singleton
class KsLog:
    def __init__(self):
        self.ks_log: logging = logging.getLogger("ks_brain")
        self.ks_log.setLevel(logging.INFO)
        # 默认日志输出控制台
        self.add_stream_log()

    def add_file_log(self, log_file):
        """
        创建文件日志(根据时间滚动, D:天, 1天一个文件, backupCount: 共保留几个文件)
        :param log_file: 日志文件名称
        """
        time_log_file = TimedRotatingFileHandler(filename=log_file, when="D", interval=1, backupCount=10, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        time_log_file.setFormatter(formatter)
        time_log_file.setLevel(logging.INFO)

        self.ks_log.addHandler(time_log_file)

    def add_stream_log(self):
        """
        添加控制台输出日志
        """
        stream_log = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        stream_log.setFormatter(formatter)
        stream_log.setLevel(logging.INFO)
        self.ks_log.addHandler(stream_log)

    def log_info(self, msg):
        self.ks_log.info(msg)

    def log_warning(self, msg):
        self.ks_log.warning(msg)



