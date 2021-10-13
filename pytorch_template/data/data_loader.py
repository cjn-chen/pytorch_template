#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   data_loader.py
@Description    :   用于加载数据
'''
from abc import ABCMeta, abstractmethod
from .get_raw_data import GetRawData
from .create_feature import CreateFeature

class ItertorData(metaclass=ABCMeta):
    """用于加载数据，计算特征，构造迭代对象
    """
    def __init__(self, raw_data_loader: GetRawData,
                 feature_builder: CreateFeature, *args, **kwargs) -> None:
        self.raw_data_loader = raw_data_loader
        self.feature_builder = feature_builder

    # 共用的流程
    def __call__(self):
        ####################################
        # 加载原始数据
        ####################################
        self.raw_data_loader()  # 读取数据
        self.raw_data = self.raw_data_loader.get_data()

    def __get_
