#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   model.py
@Description    :   构建模型的类
'''
from abc import abstractclassmethod
from torch.nn import Module


class ModelBase(Module):
    @abstractclassmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def forward():
        pass
