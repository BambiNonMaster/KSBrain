from pandas import DataFrame, Series, read_csv, read_excel
from traceback import extract_stack
from .ks_utility import virtual
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_1samp, ttest_ind, f_oneway
from typing import Callable
from logging import Logger


"""
author: mengru.du
create: 2022.3.12
modify: 20220315

该文件定义了数据处理类
1. 标准数据文件: 第一列时间, 第二列标签, 其他列特征
2. 默认数据无需预处理, 若必须预处理则重写 preprocess 方法  
3. 数据处理流程在 run 方法中实现

使用方法
1. 所有需要保存的数据都存放在 self._result 字典中
2. run(执行流程) 方法如果被继承只能重写流程, 无法扩展(没必要)
3. preprocess, transform, standardize,  analysis, select 方法可以被继承并扩展
4. transform, standardize,  analysis, select 有一个参数 data, 并且必返回 self._data
    4.1 这样方便向外扩展(比如进行 analysis 之前在外部将 label 列取 shift, 然后再传入)
5. 每次数据变形后建议调用 filldata 方法处理数据, 包括
    5.1 向下填充数据
    5.2 删除空值
    5.3 删除值全相同的列(首次)
    5.4 获取训练集分割点索引(首次)
    5.5 保存每次数据处理后的数据形状
"""


class KsProcessor:
    def __init__(self):
        self._config = None
        self._log: Logger = None
        self._data: DataFrame = DataFrame()
        self._result: dict = {}                  # 结果保存字典

        self._drop_const: bool = False           # 判断是否删除值全相同的列
        self._split_index = None                 # 记录训练集和测试集那个分割索引    

    def init(self, config=None, log: Logger = None):
        """
        数据处理对象初始化, 传入配置对象和日志对象即可
        :param config:
        :param log:
        """
        self._config = config
        self._log = log

    @property
    def get_data(self):
        """
        允许从外部获取 self._data
        :return:
        """
        return self._data

    @property
    def empty(self):
        """
        判断数据是否为空
        :return:
        """
        return self._data.empty

    @property
    def get_result(self):
        """
        允许从外部获取 self._result
        :return:
        """
        return self._result

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
    def get_split_index(self):
        """
        允许从外部获取 self._split_index
        :return:
        """
        return self._split_index

    @virtual
    def preprocess(self):
        """
        数据预处理, 默认读取数据文件, 若给的数据为非标准数据, 则重写该方法
        必须将最终处理的数据按: 第一列时间, 第二列标签, 其他列特征 的标准返回给 self._data
        """
        data_file: str = self._config["env"]["data_file"]
        if data_file.endswith("csv"):
            self._data = read_csv(data_file, index_col=0)
        elif data_file.endswith("xlsx"):
            self._data = read_excel(data_file, engine="openpyxl", index_col=0, sheet_name=0)

        self._log.info(f"原始数据: {self._data.shape}")
        
        return self._data

    def filldata(self):
        """
        # 每次处理完数据都向下填充并删除空值, 保证没有缺失值
        """
        # 1. 向下填充
        self._data.fillna(method='ffill', inplace=True)
        # 2. 删除空值
        self._data.dropna(inplace=True)

        # 3. 删除值全相同的列(只操作一次即可)
        if not self._drop_const:
            for column in self._data.columns:
                if self._data[column].isin([self._data[column][0]]).all():
                    self._data.pop(column)
            self._drop_const = True

        # 4. 获取训练集分割点索引(只操作一次即可)
        if not self._split_index:
            split = self._config["base"]["split"]
            self._split_index = self._data.index[int(self._data.shape[0] * split)]

        # 5. 获取调用该函数的函数名
        func = extract_stack()[-2][2]
        # 保存每次数据处理后的数据形状
        self._result[f"data_{func}"] = str(self._data.shape[0])

        # 输出日志
        self._log.info(f"进入 {func} 数据量: {self._data.shape}")

    @virtual
    def transform(self, data: DataFrame = None, callabck: Callable = None):
        """
        特征变换, 去极值等
        """
        if data:
            self._data = data

        self.filldata()
        
        transform = self._config["processor"].get("transform", None)
        if transform:
            # 获取所有特征列
            features = self._data.iloc[:, 1:].copy()
            log = transform.get("log", None)
            diff = transform.get("diff", None)
            # log 变形: 数据列必须全部大于0
            if log:
                log_columns: Series = (features > 0).all()
                log_columns = log_columns[log_columns.values].index
                if not log_columns.empty:
                    log_feature = features[log_columns]
                    self._data = self._data.merge(log_feature, left_index=True, right_index=True, suffixes=(None, "_log"))
            # diff 变形
            if diff:
                for d in diff:
                    diff_feature = features.diff(d)
                    self._data = self._data.merge(diff_feature, left_index=True, right_index=True, suffixes=(None, f"_diff{d}"))
                    
        return self._data

    @virtual
    def standardize(self, data: DataFrame = None):
        """
        去极值, 标准化
        """
        if data:
            self._data = data

        self.filldata()

        standardize = self._config["processor"].get("standardize", None)
        if standardize:
            # 1. 获取训练集数据
            features: DataFrame = self._data.iloc[:, 1:].copy()
            # 2. 去掉全为 0, 1的列
            std_columns = []
            for col in features.columns:
                if not features[col].isin([0, 1]).all():
                    std_columns.append(col)
            features = features[std_columns]

            # 3. 获取训练集数据
            train_data = features.loc[:self._split_index, :]
            # 4. 计算均值和方差
            avg, std = train_data.mean(), train_data.std()
            # 5. 筛选出异常值, 并用标准差填充 n_std
            n_std = standardize.get("n_std", None)
            if n_std:
                features = features.where(features.abs() < n_std * std, n_std * std, axis=1)
            # 6. 标准化
            scale = standardize.get("scale", None)
            if scale:
                features = features.apply(lambda x: (x - avg) / std, axis=1)
            # 7. 将分割点, 均值和方差都保存到字典中
            self._result["split"] = self._split_index
            self._result["avg"] = {k: v for k in features.columns for v in avg}
            self._result["std"] = {k: v for k in features.columns for v in std}
            # 8. 处理完数据后将数据再返回给 self._data
            self._data[features.columns] = features
            
        return self._data

    @virtual
    def analysis(self, data: DataFrame = None):
        """
        添加假设检验方法
        """
        if data:
            self._data = data

        self.filldata()

        # 1. 获取训练集数据
        train_data = self._data.loc[:self._split_index, :].copy()
        train_label = train_data.pop("label")

        adf_p = []          # adf 检验P_vlaue
        t_p = []            # t 检验P_vlaue
        anova_p = []       # anova 检验P_vlaue
        # adf 检验
        for col in train_data.columns:
            # 检验平稳性
            adf_result = adfuller(train_data[col], regression="c")
            adf_p.append(adf_result[1])

            # 检验两个独立样本的均值是否存在显著差异
            t_result = ttest_ind(train_data[col], train_label)
            t_p.append(t_result[1])

            # 测试两个或两个以上独立样本的均值是否存在显著差异
            anova_resutl = f_oneway(train_data[col], train_label)
            anova_p.append(anova_resutl[1])

        analysis_result = DataFrame({"p_adf": adf_p, "p_t": t_p, "p_anova": anova_p}, index=train_data.columns)
        # 保存数据检验结果
        self._result["analysis"] = analysis_result
        # 筛选因子列: 先通过 adf 检验, 然后通过 t 或 anova 检验
        level = self._config["processor"].get("level", None)
        if level:
            adf = level.get("adf", None)
            ttest = level.get("ttest", None)
            anova = level.get("anova", None)
            if adf:
                analysis_result = analysis_result[analysis_result["p_adf"] < adf]
                self._log.info(f"通过 adf 检验的因子有 {analysis_result.shape[0]} 个")
                # 保存数据检验结果
                self._result["pass_adf"] = analysis_result
            else:
                self._log.warning("缺少adf检验参数")
            if ttest and anova:
                analysis_result = analysis_result[(analysis_result["p_t"] < ttest) | (analysis_result["p_anova"] < anova)]
                self._log.info(f"通过 t/anova 检验的因子有 {analysis_result.shape[0]} 个")
                # 保存数据检验结果
                self._result["pass_t/anova"] = analysis_result
            else:
                self._log.warning("缺少 t 检验或 anova 检验参数")
        else:
            self._log.warning("没有设置假设检验参数")

        # 将 label 列和筛选的因子列都添加进来
        columns = ["label", ] + analysis_result.index.tolist()
        self._data = self._data[columns]
        return self._data

    @virtual
    def select(self, data: DataFrame = None, callabck: Callable = None):
        """
        因子筛选
        """
        if data:
            self._data = data

        self.filldata()

        select = self._config["processor"]["select"]
        if select == "stepwise":
            self._stepwise()
        elif select == "greedy":
            self._greedy()
        elif select == "lasso":
            self._lasso()
        else:
            self._data = callabck(self._data)

        self._log.info(f"最终数据: {self._data.shape}")
        return self._data

    def _stepwise(self):

        # !!! 暂时没写好, 先假设如此
        self._data = self._data.iloc[:, :10]

    def _greedy(self):
        ...

    def _lasso(self):
        ...

    @virtual
    def run(self):
        # 判断方法是否被子类重写/调用
        # func = extract_stack()[-2][2]
        # if getattr(self.__class__, func) is not getattr(KsProcessor, func):
        #     return

        # 1. 数据预处理
        self.preprocess()
        # 2. 特征变换
        self.transform()
        # 3. 标准化
        self.standardize()
        # 4. 假设检验
        self.analysis()
        # 5. 因子筛选
        self.select()
