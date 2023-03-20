#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   dataset_general.py
@Description    :   用于构造dataset（未经过数据集划分）
'''
from abc import ABCMeta, abstractmethod
from typing import Iterator
from torch.utils.data import Dataset


class Dataset_General(Dataset):
    def __init__(self, *args, **kwargs):
        """用于构造dataset方法（未经过数据集划分）
        """
        pass

    def __getitem__(self, index):
        """用于获取元素

        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        return tuple_results

    def __len__(self):
        return len(self.new_current_index)