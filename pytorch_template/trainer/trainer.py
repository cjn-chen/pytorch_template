#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   trainer.py
@Description    :   待文件用于控制程序的主要训练过程
'''
from argparse import ArgumentParser
import argparse
from ..util.construct_args import Arguments


class Trainer(object):
    """用于控制训练的主流程
    """
    def __init__(self, arguments: ArgumentParser) -> None:
        self._check_args(arguments)

    def _check_args(self, arguments: ArgumentParser) -> None:
        """检查是否具备了所有必备的属性

        Args:
            arguments (ArgumentParser): 需要检查必要属性的类
        """
        pass

    def _set_arguments(self, arguments: ArgumentParser):
        if isinstance(arguments, dict):
            return Arguments(arguments)
        else:
            raise ValueError('无法处理输入的参数类arguments的类型')

    def _set_raw_data_loader(self, raw_data_loader):
        self.raw_data_loader = raw_data_loader
