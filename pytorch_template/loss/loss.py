#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   loss.py
@Description    :   设置损失函数
'''
from abc import abstractmethod
import torch.nn as nn


class Loss(nn.Module):
    """损失函数的接口，每个损失函数都需要实现__init__方法中的__name__和forward方法
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.__name__ = "name of loss func"

    @abstractmethod
    def forward():
        pass
