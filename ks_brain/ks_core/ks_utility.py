from typing import Callable


"""
author: mengru.du
create: 2022.3.14
modify: 20220314

该文件定义了框架辅助函数
1. 单例装饰器, 日志对象需要单例模型
2. 虚函数装饰器, 父类可重写方法被标记为虚函数
"""


def singleton(cls):
    """
    单例装饰器
    :param cls:
    :return:
    """
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


def virtual(func: Callable) -> Callable:
    """
    虚函数装饰器, 在类中标记函数为虚函数, 意味着该方法需要被重写
    :param func:
    :return:
    """
    return func
