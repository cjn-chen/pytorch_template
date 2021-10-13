#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   create_feature.py
@Description    :   用于依据原始数据增加特征
'''
from abc import ABCMeta, abstractmethod
import pandas as pd


class CreateFeature(metaclass=ABCMeta):
    """用于构造特征
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, raw_data_i: pd.DataFrame) -> pd.DataFrame:
        """用于构造特征数据

        Args:
            raw_data_i (pd.DataFrame): 需要构造特征的df

        Returns:
            pd.DataFrame: 返回增加了特征列的df
        """
        pass
