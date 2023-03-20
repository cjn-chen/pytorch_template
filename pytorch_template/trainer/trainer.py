#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   trainer.py
@Description    :   待文件用于控制程序的主要训练过程
'''
# import argparse
# import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
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
            "_model",
            "_evaluators",
            "_loss",
            "_optimizer",
            "_lr_scheduler",
            "_early_stop",
        ], None)
        # 输入的参数中需要的必要参数
        self._necessary_args = dict.fromkeys(
            ['train_epochs', 'checkpoints_path'], None)

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
        # 检查必要的参数
        test_class_attrs_methods(self, self._necessary_args)

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

        # step3 训练模型
        self._train_epoches()

    def _vali(self, data_loader: DataLoader):
        """用于在每个epoch结束后，计算验证集的评价指标

        Args:
            data_loader (DataLoader): [description]

        Returns:
            evalutors_result (dict): 评价指标输出的结果，key为对应的指标的名称，
                value为对应的指标值
            vali_loss (float): data_loader中，损失函数对应的结果
        """
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            losses = []

            for i, data_tuple in enumerate(data_loader):
                pred, true = self._process_one_batch(data_tuple)
                losses.append(self._loss(pred, true).item())
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

            preds = np.array(preds)
            trues = np.array(trues)

            # 构造返回的结果
            evalutors_result = self._evaluator(preds, trues)
            loss = np.average(losses)
            # 将loss的结果存入evalutors_result中
            evalutors_result['loss'] = loss

        self.model.train()
        return evalutors_result

    ####################################
    # region 用于跟踪模型训练过程中的衡量指标
    ####################################
    def _evaluate_on_train_data_init(self) -> None:
        # 用于跟踪模型训练过程中的衡量指标
        self._evalutors_result_on_train = {}
        self._evalutors_result_on_train_time = 0

    def _evaluate_on_train_data(self, pred: torch.tensor, true: torch.tensor, loss: float) -> None:
        # 模型训练过程中，每个batch进行衡量指标的更新
        preds = np.array([pred.detach().cpu().numpy()])
        trues = np.array([true.detach().cpu().numpy()])
        evalutors_result_on_train_i = self._evaluator(preds, trues)
        for k in evalutors_result_on_train_i:
            self._evalutors_result_on_train[
                k] = self._evalutors_result_on_train.get(
                    k, 0) + evalutors_result_on_train_i[k]
            self._evalutors_result_on_train_time += 1
        evalutors_result_on_train_i['loss'] = loss

    def _evaluate_on_train_data_average(self) -> None:
        # 每个epoch结束后，对指标求平均
        result = {}
        for k in self._evalutors_result_on_train:
            result[k] = self._evalutors_result_on_train[
                k] / self._evalutors_result_on_train_time
        return result

    def _agg_evalutors(self, *dict_args):
        # 将各训练集的结果进行聚合，合并为一个pd.DataFrame
        

    # endregion



    def _process_on_batch(self, data_tuple, *args, **kwargs):
        pass

    def _train_epoches(self, train_loader, valid_loader, test_loader, *args,
                       **kwargs):
        """主要模型训练过程

        Args:
            data_loader ([type]): torch的DataLoader
        """
        if self.arguments.use_amp:
            # 是否使用automatic mixed precision，即在梯度
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.arguments.train_epochs):
            self.model.train()
            # 用于跟踪模型训练过程中的衡量指标
            self._evaluate_on_train_data_init()
            for i, data_tuple in enumerate(train_loader):
                # 清空梯度
                self._optimizer.zero_grad()

                ####################################
                # forward
                ####################################
                pred, true = self._process_on_batch(data_tuple)

                ####################################
                # 计算loss
                ####################################
                loss = self._loss(pred, true)

                ####################################
                # 计算需要跟踪的指标
                ####################################
                self._evaluate_on_train_data(pred, true, loss.item())

                ####################################
                # backward
                ####################################
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self._optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self._optimizer.step()

            ####################################
            # 计算并记录需要的指标
            ####################################
            train_evalutors_result = self._evaluate_on_train_data_average()
            vali_evalutors_result = self._vali(valid_loader)
            test_evalutors_result = self._vali(test_loader)

            ####################################
            # early stop 以及学习率调整
            ####################################
            self._early_stop(vali_evalutors_result['loss'], self.model, self.checkpoints_path)

            ####################################
            # 自定义的scheduler
            ####################################
            self._lr_scheduler(self._optimizer, epoch + 1, self.arguments)

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

    def set_evaluator(self, evaluator, *args, **kwargs):
        self._evaluator = evaluator

    def set_loss(self, loss, *args, **kwargs):
        self._loss = loss

    def set_optim(self, optimizer, *args, **kwargs):
        self._optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler, *args, **kwargs):
        self._lr_scheduler = lr_scheduler

    def set_early_stop(self, early_stop, *args, **kwargs):
        self._early_stop = early_stop

    # endregion
