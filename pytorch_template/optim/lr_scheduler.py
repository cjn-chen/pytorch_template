#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   lr_scheduler.py
@Description    :   学习率调整类
'''
from abc import ABCMeta, abstractmethod
from torch.optim import Optimizer


class LrScheduler(metaclass=ABCMeta):
    def __init__(self, optimizer: Optimizer, *args, **kwargs) -> None:
        """接收优化器，用于控制其中的学习了参数

        Args:
            optimizer (Optimizer): 用于控制梯度的变化
        """
        self.optimizer = optimizer  # 优化器
        self.epoch = 0  # 记录目前学习的epoch

    @abstractmethod
    def step(self):
        """ 控制优化器
        """
        self.epoch = self.epoch + 1  # 记录目前学习的epoch

    def update_lr_in_optimizer(self, new_lr: float):
        """更新optim中的学习率

        Args:
            new_lr (float): 将optim中的学习率更新为new_lr
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class AdjustLearningRate(LrScheduler):
    def __init__(self, optimizer: Optimizer, *args, **kwargs) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.kwargs = kwargs

    def step(self):
        super().step()
        if self.kwargs.lradj == 'type1':
            lr_adjust = {
                self.epoch:
                self.kwargs.learning_rate * (0.5**((self.epoch - 1) // 2))
            }
        elif self.kwargs.lradj == 'type2':
            lr_adjust = {
                2: 5e-5,
                4: 1e-5,
                6: 5e-6,
                8: 1e-6,
                10: 5e-7,
                15: 1e-7,
                20: 5e-8
            }
        if self.epoch in lr_adjust.keys():
            lr_new = lr_adjust[self.epoch]
            self.update_lr_in_optimizer(lr_new)
