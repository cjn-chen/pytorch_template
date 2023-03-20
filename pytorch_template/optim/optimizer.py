#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   optimizer.py
@Description    :   用于自定义optimer，进行控制梯度的更新
'''
from abc import abstractmethod
from torch.optim import Optimizer


class OptimizerCustom(Optimizer):
    """ optim的抽象类（接口类）
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self):
        """ 控制优化器
        """
        self.optimizer.step()
