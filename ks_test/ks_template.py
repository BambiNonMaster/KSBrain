import pandas as pd
from pandas import DataFrame
from ks_brain import (
    KsBrain,
    KsProcessor,
    KsModel,
    KsRegressor,
    KsClassifier,
    KsUnivariate,
    KsMultivariate
)

"""
author: mengru.du
create: 2022.3.12
modify: 20220315

该文件提供了 ks_brain 的使用模板
1. 重写对象方法
    1. Processor 类继承 KsProcessor 自定义因子筛选功能
    2. Model1 类继承 KsRegressor 自定义回归模型预测功能
2. 自定义执行流程
    1. 继承 KsBrain 首先初始化日志对象
    2. 使用 add_processor 添加因子筛选对象
    3. 使用 add_model 添加模型对象
    5. 重写 run 方法, 自定义程序执行流程
"""


pd.set_option("display.max_columns", 2000)
pd.set_option("display.width", 2000)


# 重写数据处理
class Processor(KsProcessor):

    # def select(self, data: DataFrame = None):
    #     """
    #     由于有三种因子筛选的方法, 因此这里需要重写
    #     """

    def run(self):
        active = self.get_config["processor"].get("active", None)
        if active:
            # 1. 数据预处理
            self.preprocess()
            # 2. 特征变换
            self.transform()
            # 3. 去极值
            self.standardize()
            # 4. 假设检验
            self.analysis()
            # 5. 因子挑选
            self.select()


# 重写模型
class Model1(KsRegressor):
    ...


# 重写模型
class Model2(KsClassifier):
    ...


# 重写模型
class Model3(KsUnivariate):
    ...


# 重写模型
class Model4(KsMultivariate):
    ...


# 重写运行逻辑
class Brain(KsBrain):
    def __init__(self, config: str):
        super(Brain, self).__init__()
        # 初始化日志文件
        self.init(config)

    def add_processor(self, processor: KsProcessor):
        """
        添加数据处理对象
        :param processor:
        """
        self.register_processor(processor)

    def add_model(self, model: KsModel):
        """
        添加模型对象
        :param model:
        """
        self.register_model(model)

    def run(self):
        """
        重写程序执行流程
        """
        super(Brain, self).run()



if __name__ == '__main__':
    config_file = "../ks_test/config_eurusd.yaml"
    brain = Brain(config_file)
    brain.add_processor(Processor())
    brain.add_model(Model1())
    brain.add_model(Model2())
    # brain.add_model(Model3())
    # brain.add_model(Model4())
    brain.run()
