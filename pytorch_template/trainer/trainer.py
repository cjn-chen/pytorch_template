#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   trainer.py
@Description    :   待文件用于控制程序的主要训练过程
'''
import argparse
import torch
from argparse import ArgumentParser
from ..util.utils import test_class_attrs_methods
from ..log.logger import LoggerHandler

logger = LoggerHandler()


class Trainer(object):
    """用于控制训练的主流程
    """
    def __init__(self, arguments: ArgumentParser) -> None:
        # 检查参数
        self._test_args(arguments)
        self.arguments = arguments

        # 需要检查的关键组件及其需要实现的方法
        self._critical_components_check_list = dict.fromkeys([
            "_model", "_evaluators", "_loss", "_optimizer", "_lr_scheduler",
            "_early_stop"
        ], None)

    ####################################
    # region 模型中用于检测各个环节是否满足所需要的属性方法
    ####################################
    def _test_args(self, arguments: ArgumentParser) -> None:
        """检查是否具备了所有必备的属性，出错误的话会raise error

        Args:
            arguments (ArgumentParser): 需要检查必要属性的类
        """
        # 检查checkpoint_path是否合法
        if self.arguments.checkpoint_path is not None:
            if not isinstance(self.arguments.checkpoint_path, str):
                raise ValueError("checkpoint_path参数必须是字符串")

    # endregion

    ####################################
    # region 控制流程
    ####################################
    def train(self, *args, **kwargs):
        """模型的训练过程
        """
        # step1 构造训练需要的关键部件
        self._prepare_critical_components()

        # step2 检查是否有checkpoint，有的话需要进行模型加载
        self._load_model_checkpoint()

    def _train_batch(self, *args, **kwargs):
        pass

    def _train_epoches(self, *args, **kwargs):
        pass

    # endregion

    ####################################
    # region 加载模型的checkpoint
    ####################################
    def _load_model_checkpoint(self, *args, **kwargs):
        if self.arguments.checkpoint_path is not None:
            # 如果包含checkpoint
            self.model.load_state_dict(
                torch.load(self.arguments.checkpoint_path))

    # endregion

    ####################################
    # 核心组件的构造和组装
    ####################################
    # region 核心组件构建，optiom，loss，lr_scheduler，early_stop等训练过程中
    def _prepare_critical_components(self, *args, **kwargs):
        # 检查是否需要的属性和类都有
        test_class_attrs_methods(self, self._critical_components_check_list)

    def set_model(self, model, *args, **kwargs):
        self._model = model

    def set_evaluators(self, evaluators, *args, **kwargs):
        if not isinstance(evaluators, list):
            self._evaluators = [evaluators]

    def set_loss(self, loss, *args, **kwargs):
        self._loss = loss

    def set_optim(self, optimizer, *args, **kwargs):
        self._optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler, *args, **kwargs):
        self._lr_scheduler = lr_scheduler

    def set_early_stop(self, early_stop, *args, **kwargs):
        self._early_stop = early_stop

    # endregion
