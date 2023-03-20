from abc import abstractmethod, abstractproperty

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   evaluator.py
@Description    :   用于评估模型训练效果的指标
'''
from abc import abstractproperty, abstractmethod
import pandas as pd
from numpy import ndarray


class EvaluatorItem():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def result(self):
        return self._result

    @abstractmethod
    def __call__(self, preds: ndarray, trues: ndarray):
        # 计算损失，将结果保存在self._result
        # 比如：self._result = {'valuator1': 10, 'evaluator2': 95}
        pass


class Evaluators():
    def __init__(self, evalutors: list) -> None:
        """用于计算模型结果的效果，计算验证集的效果

        Args:
            evalutors (list of EvaluatorItem): 由EvaluatorItem构成的list
        """
        self.evalutors = evalutors

    def result(self):
        return self._result

    def __call__(self, preds: ndarray, trues: ndarray):
        # 计算损失，将结果保存在self._result
        # 比如：self._result = {'valuator1': 10, 'evaluator2': 95}
        self._result = {}
        for evalutor_i in self.evalutors:
            result_i = evalutor_i(preds, trues)
            self._result.update(result_i)