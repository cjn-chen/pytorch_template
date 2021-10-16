#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   early_stopping.py
@Description    :   用于控制早停机制
'''
import torch
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from ..log.logger import LoggerHandler

logger = LoggerHandler()


class EarlyStoppingInterface(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, val_loss: float, model: torch.nn.Module, path: str):
        """用于判断是否符合早停的条件

        Args:
            val_loss (float): 验证集损失
            model (torch.nn.Module): 模型，用于保存模型的参数
            path (str): 保存checkpoint的路径
        """
        pass

    @abstractproperty
    def early_stop(self):
        return self._early_stop

    def save_checkpoint(
        self,
        val_loss: float,
        model: torch.nn.Module,
        path: str,
    ):
        """用于保存模型

        Args:
            val_loss (float): 验证集损失
            model (torch.nn.Module): 模型，用于保存模型的参数
            path (str): 保存checkpoint的路径
        """
        # 用于保存最优结果的函数
        if self.verbose:
            logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')


class EarlyStopping(EarlyStoppingInterface):
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = True,
        delta: float = 0,
    ) -> None:
        """用于控制训练流程中的早停机制

        Args:
            patience (int, optional): 连续几步的valid set中的表现没有优化的话则发出停止训练的信号. Defaults to 7.
            verbose (bool, optional): 是否输出log. Defaults to True.
            delta (float, optional): 判断是否最优的时候，loss相比之下升高了多少则认为没有改进. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # 用于计数，判断累计几次没有优化
        self.best_score = None  # 记录其中最好的结果
        self._early_stop = False  # 是否进行早停
        self.delta = delta  # 判断是否最优的时候，辅助判断的参数

    def early_stop(self):
        return self._early_stop

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str):
        """用于判断是否符合早停的条件

        Args:
            val_loss (float): 验证集损失
            model (torch.nn.Module): 模型，用于保存模型的参数
            path (str): 保存checkpoint的路径
        """
        # 记录最佳loss，越小越好
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_val_loss + self.delta:
            # 记录loss连续几次没有达到最优结果
            self.counter += 1
            if self.verbose:
                logger.info(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            # 如果超过self.patience次没有取得最优结果，则提示进行早停
            if self.counter >= self.patience:
                self._early_stop = True
        else:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
