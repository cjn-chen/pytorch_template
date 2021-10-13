#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   construct_args.py
@Description    :   用于将字典类型转换为类，调用方式和ArgumentParser保持一致
'''


class Arguments():
    """可以将字典转换为类，字典中的key都变成属性，value都变成
    """
    def __init__(self, args_dict: dict) -> None:
        self.args_dict = args_dict

    def __getattr__(self, item):
        return self.args_dict.get(item, None)


if __name__ == "__main__":
    args_dict = {
        'name': 'test',
        'optim': 'rmsprop',
    }
    args = Arguments(args_dict)
    print(args.name)
    print('name' in args.name)
