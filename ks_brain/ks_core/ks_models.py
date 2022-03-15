from abc import ABCMeta, abstractmethod
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostRegressor,
    AdaBoostClassifier,
    VotingClassifier,
    VotingRegressor
)
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier
)
from lightgbm import (
    LGBMRegressor,
    LGBMClassifier
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.svar_model import SVAR

from sklearn.preprocessing import OneHotEncoder
from logging import Logger
from pandas import DataFrame, read_excel, read_csv
import sys
from .ks_utility import virtual
from numpy import where



"""
author: mengru.du
create: 2022.3.12
modify: 20220315

该文件定义了模型类 
1. 机器学习模型包括分类和回归模型, 只可以做值预测
2. 计量经济学模型包括单变量模型和多变量模型, 可以做值预测和路径预测
"""


class KsModel(metaclass=ABCMeta):
    def __init__(self):
        self._config = None                      # 配置内容
        self._log: Logger = None               # 日志对象
        self._data = None                        # 基本数据
        self._result: dict = {}                  # 结果保存对象

        self._split_index = None                 # 训练集分割索引

    def init(self, config = None, log: Logger = None, kw: dict = None):
        """
        :param config: 配置
        :param log: 日志对象
        :param kw: 通过因子筛选传入后需要传入的参数
        """
        self._config = config
        self._log = log

        # 如果正确传入数据, (可能先走了因子筛选流程)
        if kw:
            self._data = kw["data"]
            self._split_index = kw["split_index"]
        # 如果没有传入数据, 直接通过配置文件读取文件数据
        else:
            data_file: str = self._config["env"]["data_file"]
            if data_file.endswith("csv"):
                self._data = read_csv(data_file, index_col=0)
            elif data_file.endswith("xlsx"):
                self._data = read_excel(data_file, engine="openpyxl", index_col=0, sheet_name=0)
            else:
                self._log.warning("模型没有数据!!!")
                return

            split = self._config["base"]["split"]
            self._split_index = self._data.index[int(self._data.shape[0] * split)]
            self._log.info(f"模型从文件加载原始数据: {self._data.shape}")
        self._result["raw_data"] = self._data.shape[0]

    @property
    def get_config(self):
        """
        为子类提供获取 self._config
        :return:
        """
        return self._config

    @property
    def get_log(self):
        """
        为子类提供获取 self._log
        :return:
        """
        return self._log

    @property
    def get_data(self):
        """
        允许从外部获取 self._data
        :return:
        """
        return self._data

    @property
    def get_result(self):
        """
        允许从外部获取 self._result
        :return:
        """
        return self._result

    @property
    def split_index(self):
        """
        允许从外部获取 self._split_index
        :return:
        """
        return self._split_index

    @abstractmethod
    def predict_path(self):
        # 路径预测
        ...

    @abstractmethod
    def predict_value(self):
        # 值预测
        ...

    # 模型执行流程, 需要子类重写
    @abstractmethod
    def build_models(self):
        # 根据配置确定, 构造模型
        ...

    # 模型执行流程, 需要子类重写
    @abstractmethod
    def run(self):
        # 运行逻辑
        ...


class KsClassifier(KsModel):
    def __init__(self):
        super(KsClassifier, self).__init__()
        self._models_list: list = []                    # 保存模型列表
        self._models_reuslt: DataFrame = DataFrame()    # 保存模型预测结果(第一列是时间, 第二列是真实值 true, 之后是模型预测值)

    def init(self, config=None, log: Logger = None, kw: dict = None):
        """
        分类模型初始化是首先需要将标签分类
        :param config: 配置对象
        :param log: 日志对象
        :param kw: 参数
        :return: 返回值
        """
        super(KsClassifier, self).init(config, log, kw)
        if (self._data["label"] > 0).all() or (self._data["label"] < 0).all():
            self._log.warning("label列方向一致, 无法计算准确率")
            return

        # 
        self._data["label"] = where(self._data["label"] > 0, 1, 0)

    @virtual
    def predict_path(self):
        # 路径预测
        self.get_log.warning("分类模型无法进行路径预测")

    @virtual
    def predict_value(self):
        # 值预测, 即标签预测
        if self._data.empty or not self.split_index:
            self._log.warning("模型初始化未成功, 缺少数据或分割索引")
            return

        if len(self._models_list) == 0:
            self._log.warning("没有模型, 无法预测")
            return

        base = self._config["models"]["base"]
        mode = base.get("mode", True)  # 默认循环训练
        freq = base.get("freq", 5)  # 默5期重新训练
        window = base.get("window", 100)  # 默每次用最近100条数据
        value = base.get("value", 1)  # 默认每次预测未来1期数据

        data = self._data
        spx = data.index.to_list().index(self.split_index)
        # 保存索引位置数据
        self._result["split"] = spx

        # 保存训练结果
        df_result: DataFrame = DataFrame()
        if mode:
            # 记录连续预测了多少次
            freq_count = 0
            # 保存所有模型预测结果
            items = []
            for sl in range(spx, data.shape[0] - value):  # 因为预测未来n期, 因此这里 -value
                freq_count += 1
                train_data = data.iloc[sl - window:sl, 1:]
                train_label = data.iloc[sl - window + value:sl + value, 0]  # 对应训练集 label +1

                test_data = data.iloc[sl:sl + 1, 1:]
                test_label = data.iloc[sl + value:sl + value + 1, 0]  # 对应测试集 label +1

                item = {}
                item["true"] = test_label.values[0]
                for model in self._models_list:
                    if freq_count == 1:
                        model.fit(train_data.values, train_label.values)
                    if freq_count > freq:
                        freq_count = 0
                    pred = model.predict(test_data.values)
                    item[model.__class__.__name__] = pred[0]  # 循环预测每次只有一个值
                items.append(item)
            df_result = DataFrame(items, index=data.index[spx + value:])
        else:
            label = data.pop("label")
            train_data = data.iloc[:spx, :]
            train_label = label[value:spx + value]

            test_data = data.iloc[spx:-value, :]
            test_label = label[spx + value:]

            item = {}
            item["true"] = test_label.values
            for model in self._models_list:
                model.fit(train_data.values, train_label.values)
                pred = model.predict(test_data.values)

                item[model.__class__.__name__] = pred  # 非循环预测一次有多个值
            df_result = DataFrame(item, index=data.index[spx + value:])

        # 将结果保存起来
        self._models_reuslt = df_result
        # 保存预测结果 (类名 + 函数名)
        self._result[f"{self.__str__()}_{sys._getframe().f_code.co_name}"] = df_result

    def evaluate(self):
        # 模型评估, 判断方向准确率
        assert not self._models_reuslt.empty, "模型训练结果为空"

        df_result = self._models_reuslt
        for col in df_result.columns[1:]:
            # 分类没有误差值(结果第一列是真实值)
            self._result[f"accuracy_{col}"] = accuracy_score(df_result["true"], df_result[col])
            self._result[f"recall_{col}"] = recall_score(df_result["true"], df_result[col])
            self._result[f"f1_{col}"] = f1_score(df_result["true"], df_result[col])

            self._log.info(f"{col} 准确率: {self._result[f'accuracy_{col}']}")

    # 模型执行流程, 需要子类重写
    @virtual
    def build_models(self):
        active = self._config["models"]["classifier"].get("active", None)
        ensemble = self._config["models"]["classifier"].get("ensemble", None)
        models = self._config["models"]["classifier"].get("models", None)
        # 组合模型评估器
        estimators = []
        if active and models:
            for name in models:
                params: dict = models[name].get("params", None)
                if name == "LogisticRegression":
                    if params:
                        model = LogisticRegression(**params)
                    else:
                        model = LogisticRegression()
                elif name == "RandomForestClassifier":
                    if params:
                        model = RandomForestClassifier(**params)
                    else:
                        model = RandomForestClassifier()
                else:
                    model = None
                    self.get_log.warning(f"框架中不包含模型: {name}")
                    continue
                # 将模型保存到组合模型评估器 和 模型列表
                estimators.append((name, model))
                self._models_list.append(model)

            if ensemble:
                weights = ensemble.get("weights", None)
                voting = ensemble.get("voting", "hard")     # 默认硬投票
                ensemble = VotingClassifier(estimators=estimators, voting=voting, weights=weights)
                # 将组合模型添加到模型列表
                self._models_list.append(ensemble)
            else:
                self._log.warning("没有添加组合模型")
                
    # 模型执行流程, 需要子类重写
    @virtual
    def run(self):
        self.build_models()
        self.predict_value()
        self.evaluate()

    def __str__(self):
        return "classifier"


class KsRegressor(KsModel):
    def __init__(self):
        super(KsRegressor, self).__init__()

        self._models_list: list = []                    # 保存模型列表
        self._models_reuslt: DataFrame = DataFrame()    # 保存模型预测结果(第一列是时间, 第二列是真实值 true, 之后是模型预测值)

    def init(self, config=None, log: Logger = None, kw: dict = None):
        super(KsRegressor, self).init(config, log, kw)
        if (self._data["label"] > 0).all() or (self._data["label"] < 0).all():
            self._log.warning("label列方向一致, 无法计算准确率")
            return

    @virtual
    def predict_path(self):
        # 路径预测
        self.get_log.warning("回归模型无法进行路径预测")

    @virtual
    def predict_value(self):
        # 值预测
        if self._data.empty or not self.split_index:
            self._log.warning("模型初始化未成功, 缺少数据或分割索引")
            return

        if len(self._models_list) == 0:
            self._log.warning("没有模型, 无法预测")
            return

        base = self._config["models"]["base"]
        mode = base.get("mode", True)               # 默认循环训练
        freq = base.get("freq", 5)                  # 默5期重新训练
        window = base.get("window", 100)            # 默每次用最近100条数据
        value = base.get("value", 1)                # 默认每次预测未来1期数据

        data = self._data
        spx = data.index.to_list().index(self.split_index)
        # 保存索引位置数据
        self._result["split"] = spx

        # 保存训练结果
        df_result: DataFrame = DataFrame()
        if mode:
            # 记录连续预测了多少次
            freq_count = 0
            # 保存所有模型预测结果
            items = []
            for sl in range(spx, data.shape[0] - value):                # 因为预测未来n期, 因此这里 -value
                freq_count += 1
                train_data = data.iloc[sl-window:sl, 1:]
                train_label = data.iloc[sl-window+value:sl+value, 0]    # 对应训练集 label +1

                test_data = data.iloc[sl:sl+1, 1:]
                test_label = data.iloc[sl+value:sl+value+1, 0]          # 对应测试集 label +1

                item = {}
                item["true"] = test_label.values[0]
                for model in self._models_list:
                    if freq_count == 1:
                        model.fit(train_data.values, train_label.values)
                    if freq_count > freq:
                        freq_count = 0
                    pred = model.predict(test_data.values)
                    item[model.__class__.__name__] = pred[0]            # 循环预测每次只有一个值
                items.append(item)
            df_result = DataFrame(items, index=data.index[spx+value:])
        else:
            label = data.pop("label")
            train_data = data.iloc[:spx, :]
            train_label = label[value:spx+value]

            test_data = data.iloc[spx:-value, :]
            test_label = label[spx+value:]

            item = {}
            item["true"] = test_label.values
            for model in self._models_list:
                model.fit(train_data.values, train_label.values)
                pred = model.predict(test_data.values)

                item[model.__class__.__name__] = pred                   # 非循环预测一次有多个值
            df_result = DataFrame(item, index=data.index[spx+value:])
        # 将结果保存起来
        self._models_reuslt = df_result
        # 保存预测结果 (类名 + 函数名)
        self._result[f"{self.__str__()}_{sys._getframe().f_code.co_name}"] = df_result

    def evaluate(self):
        # 模型评估, 判断方向准确率
        assert not self._models_reuslt.empty, "模型训练结果为空"

        df_result = self._models_reuslt
        for col in df_result.columns[1:]:
            # 保存误差值(结果第一列是真实值)
            self._result[f"mse_{col}"] = round(mean_squared_error(df_result["true"], df_result[col]), 4)
            self._result[f"mae_{col}"] = round(mean_absolute_error(df_result["true"], df_result[col]), 4)
            # 计算方向准确率
            se_temp = df_result["true"] * df_result[col]
            self._result[f"rate_{col}"] = round(se_temp[se_temp > 0].shape[0] / se_temp.shape[0], 4)
            self._log.info(f"{col} 准确率: {self._result[f'rate_{col}']}")

    # 模型执行流程, 需要子类重写
    @virtual
    def build_models(self):
        active = self._config["models"]["regressor"].get("active", None)
        ensemble = self._config["models"]["regressor"].get("ensemble", None)
        models = self._config["models"]["regressor"].get("models", None)
        # 组合模型评估器
        estimators = []
        if active and models:
            for name in models:
                params: dict = models[name].get("params", None)
                if name == "LinearRegression":
                    if params:
                        model = LinearRegression(**params)
                    else:
                        model = LinearRegression()
                elif name == "RandomForestRegressor":
                    if params:
                        model = RandomForestRegressor(**params)
                    else:
                        model = RandomForestRegressor()
                else:
                    model = None
                    self.get_log.warning(f"框架中不包含模型: {name}")
                    continue
                # 将模型保存到组合模型评估器 和 模型列表
                estimators.append((name, model))
                self._models_list.append(model)

            if ensemble:
                weights = ensemble.get("weights", None)
                ensemble = VotingRegressor(estimators=estimators, weights=weights)
                # 将组合模型添加到模型列表
                self._models_list.append(ensemble)
            else:
                self._log.warning("没有添加组合模型")

    def run(self):
        self.build_models()
        self.predict_value()
        self.evaluate()

    def __str__(self):
        return "regressor"


class KsUnivariate(KsModel):

    @virtual
    def predict_path(self):
        # 路径预测
        ...

    @virtual
    def predict_value(self):
        # 值预测
        ...

    # 模型执行流程, 需要子类重写
    @virtual
    def run(self):
        # 根据配置确定， 是路径预测还是值预测
        active = self._config["models"]["univariate"].get("active", None)
        if active:
            ...

    def __str__(self):
        return "univariate"


class KsMultivariate(KsModel):

    @virtual
    def predict_path(self):
        # 路径预测
        ...

    @virtual
    def predict_value(self):
        # 值预测
        ...

    # 模型执行流程, 需要子类重写
    @virtual
    def run(self):
        # 根据配置确定， 是路径预测还是值预测
        active = self._config["models"]["multivariate"].get("active", None)
        if active:
            ...

    def __str__(self):
        return "multivariate"

