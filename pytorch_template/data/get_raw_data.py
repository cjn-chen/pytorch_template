#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   get_raw_data.py
@Description    :   用于获取原始数据
'''
from abc import ABCMeta, abstractmethod
from typing import Iterator
import pandas as pd


class GetRawData(metaclass=ABCMeta):
    """用于读取数据原始数据
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """用于读取数据
        """
        pass

    @abstractmethod
    def get_data() -> pd.DataFrame or Iterator:
        """获取读取的原始数据的结果

        Returns:
            pd.DataFrame or Iterator: 
        """
        pass