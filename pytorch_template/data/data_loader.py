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
import pandas as pd
from torch.utils.data import Dataset


####################################
# data workflow step1
####################################
class ItertorData():
    """用于加载数据，计算特征，构造迭代对象
    """
    def __init__(self, raw_data_loader: GetRawData,
                 feature_builder: CreateFeature, *args, **kwargs) -> None:
        self.raw_data_loader = raw_data_loader
        self.feature_builder = feature_builder

    # 共用的流程
    def __call__(self):
        """加载原始数据
        """
        self.raw_data_loader()  # 读取数据
        self.raw_data_iter = self.raw_data_loader.get_data()

    def __iter__(self, index):
        """返回由pandas.DataFrame构造的迭代器
        """
        # 如果使用for循环调用ItertorData，会先访问iter，再访问next
        return self

    def __next__(self):
        """每次循环的时候，进行特征构造
        """
        for raw_data_i in self.raw_data_iter:
            raw_data_with_feature = self.feature_builder(raw_data_i)
        return raw_data_with_feature


####################################
# data workflow step2
####################################
class DatasetSampler(Dataset):
    """用于构造不区分数据集的dataset
    """
    def __init__(
        self,
        raw_data_with_feature: pd.DataFrame,
        *args,
        **kwargs,
    ) -> None:
        self.raw_data_with_feature = raw_data_with_feature

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


####################################
# data workflow step3
####################################
class DatasetSplit(Dataset):
    """根据指定的数据集类型，获取对应的子集的dataset
    """
    @abstractmethod
    def __init__(
        self,
        dataset: Dataset,
        flag: str,
        *args,
        **kwargs,
    ):
        """负责接收用户自定义的dataset，根据不同的数据集进行关于时间的划分，
            将已有的dataset关于训练集进行划分

            Args:
                dataset (torch.utils.data.Dataset): 数据集
                flag (['train','test', 'val', 'pred]): 说明数据集类型，训练集、测试集或者验证集
                args (来自包argparse): 所有输入的超参数
            """
        self.dataset = dataset

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass
