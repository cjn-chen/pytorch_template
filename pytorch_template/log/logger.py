#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   chenjiannan
@File    :   log.py
@Description    :   用于输出日志
'''
import logging


class LoggerHandler(logging.Logger):
    def __new__(cls, *args, **kwargs):
        """ 修改为单例模式
        """
        if not hasattr(LoggerHandler, "_isinstance"):
            cls._isinstance = object.__new__(cls)  # 构建新对象
            cls._isinit = False  # 是否已经进行了初始化
        return cls._isinstance

    def __init__(
        self,
        name: str = "root",
        level: str = logging.DEBUG,
        file: str = None,
        format:
        str = "%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s",
    ) -> None:
        """用于构造日志生成器

        Args:
            name (str, optional): 日志的名称. Defaults to "root".
            level (str, optional): 日志的级别，超过该级别的日志才会被输出. Defaults to "DEBUG".
            file (str, optional): 输出日志的文件的名称. Defaults to None.
            format (str, optional): 日志格式. Defaults to "%(filename)s:%(lineno)d:%(name)s:%(levelname)s:%(message)s".
        """
        if not LoggerHandler._isinit:
            # logger(name)  直接超继承logger当中的name
            super().__init__(name)

            # 设置收集器级别
            # logger.setLevel(level)
            self.setLevel(level)  # 继承了Logger 返回的实例就是自己

            # 初始化format，设置格式
            fmt = logging.Formatter(format)

            # 初始化处理器
            # 如果file为空，就执行stream_handler,如果有，两个都执行
            if file:
                file_handler = logging.FileHandler(file)
                # 设置handler级别
                file_handler.setLevel(level)
                # 添加handler
                self.addHandler(file_handler)
                # 添加日志处理器
                file_handler.setFormatter(fmt)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            self.addHandler(stream_handler)
            stream_handler.setFormatter(fmt)
            LoggerHandler._isinit = True


if __name__ == '__main__':
    logger = LoggerHandler("log1", logging.DEBUG, file="demo.txt")
    logger2 = LoggerHandler()
    logger.debug("debug_msg")
    logger.info("info_msg")
    logger.warning("warning_msg")
    logger.error("error_msg")
    logger.critical("critical_msg")
