from os.path import abspath, dirname, exists
from os import chdir, mkdir
from sys import argv
from yaml import safe_load
from pandas import DataFrame
from logging import Logger
from .ks_processor import KsProcessor
from .ks_models import KsModel
from typing import List, Dict
from .ks_log import KsLog


"""
author: mengru.du
create: 2022.3.12
modify: 20220315

该文件定义程序执行的基本框架类
1. 通过 init 方法初始化 ksBrain 对象
2. 初始化时自动设置文件目录和日志
3. 通过调用 register_ 相关方法添加因子筛选和模型预测功能
4. 因子筛选只有一个, 预测模型可有多个
4. run 方法运行整个流程
"""


class KsBrain:
    def __init__(self):
        self._config = None                             # 配置文件对象
        self._storage: str = ""                         # 数据保存目录
        self._log: Logger = None                        # 日志对象
        self._results: dict = {}                        # 结果保存对象
        self.processor: KsProcessor = KsProcessor()     # 数据处理对象(一个)
        self.models: List[KsModel] = []                 # 模型对象列表(多个)

    def init(self, config_file: str):
        """
        环境准备
        """
        # 1. 将主脚本路径设置为工作路径
        chdir(dirname(abspath(argv[0])))
        # 2. 加载配置文件
        self._config = self.load_config(config_file)

        # 3. 将数据文件目录设置为工作目录
        env = self._config["env"]
        if env.get("data_file", None):
            chdir(dirname(abspath(env["data_file"])))
        else:
            self._log.warning("数据文件不存在")
            return

        # 4. 添加数据保存目录或默认 ks_storage
        if env.get("result_dir", None):
            self._storage = env["result_dir"]
        else:
            self._storage = "ks_storage"
        if not exists(self._storage):
            mkdir(self._storage)

        # 5. 设置日志
        log = KsLog()
        log_file = env.get("log_file", None)
        if log_file:
            log_file = self._storage + "/" + log_file
            log.add_file_log(log_file)
            self._log = log.get_log

    @staticmethod
    def load_config(file_name: str):
        """
        加载配置文件
        :param file_name: 配置文件名称
        """
        assert file_name.endswith("yaml"), "配置文件格式不对"
        with open(file_name, "r", encoding="utf-8") as f:
            config = safe_load(f.read())
        return config

    @property
    def get_config(self):
        return self._config

    @property
    def get_log(self):
        return self._log

    def register_processor(self, processor: KsProcessor):
        """
        添加数据处理器
        """
        # 1. 利用配置和日志初始化, 在赋值给 self.processor
        self.processor = processor

    def register_model(self, model: KsModel):
        """
        添加模型对象, 因为有多个模型可能处理 data, 因此这里的 data 需要复制
        """
        # 将模型添加到模型列表
        self.models.append(model)

    def run(self):
        """
        程序执行流程
        """
        # 模型参数
        kw = {}

        # 1. 因子筛选流程
        processor_active = self._config["processor"].get("active", None)
        if processor_active:
            self.processor.init(config=self.get_config, log=self.get_log)
            self.processor.run()
            # 1.1. 保存因子筛选数据
            self._results[type(self.processor).__base__.__name__] = self.processor.get_result
            # 1.2 如果因子筛选为空, 报错
            if self.processor.empty:
                self._log.warning("因子筛选数据为空")
                return
            else:
                # 如果因子筛选成功, 利用因子筛选结果初始化预测模型
                kw = {"data": self.processor.get_data, "split_index": self.processor.get_split_index}

        # 2. 模型预测
        for model in self.models:
            model_active = self._config["models"][model.__str__()].get("active", None)
            if model_active:
                model.init(config=self.get_config, log=self.get_log, kw=kw)
                model.run()
                # 2.1 保存模型数据
                self._results[type(model).__base__.__name__] = model.get_result
        print(self._results.keys())
