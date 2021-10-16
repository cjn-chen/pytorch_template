#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   utils.py
@Description    :   用于存放一些常用的工具
'''


def test_class_attrs_methods(cls2check: object, check_dict: dict):
    """用于检查制定的类中是否具有需要的方法和属性是否具有

    Args:
        cls2check (object): 需要进行检查的类
        check_dict (dict): 需要检查的属性和方法的说明
        {'_model': [], '_loss': ['forward']}表示需要检查_model属性、_loss.forward
        是否存在。
    """
    for attr_i in check_dict:
        if hasattr(cls2check, attr_i):
            attr_i_instance = getattr(cls2check, attr_i)
            if check_dict[attr_i] is not None:
                for attr_j in check_dict[attr_i]:
                    if not hasattr(attr_i_instance, attr_j):
                        raise ValueError(
                            f'{str(attr_i_instance)}没有需要的属性或方法{attr_j}')
        else:
            raise ValueError(f'{str(cls2check)}没有需要的属性或方法{attr_i}')


if __name__ == "__main__":
    pass