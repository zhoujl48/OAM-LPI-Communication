#!/usr/bin/python
# -*- coding:utf-8 -*-
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The Common Tools Project
"""
Project *** -- Module ***

Usage: 
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019-03-27 
"""
import numpy as np


class BER(object):
    def __init__(self, confusion_matrix, num_class=16, num_one_class=1000):
        self._confusion_matrix = confusion_matrix
        self._num_class = num_class
        self._num_one_class = num_one_class

    @property
    def _symbols(self):
        """生成码元集合
        """
        symbols = [str(bin(label))[2:] for label in range(self._num_class)]
        symbols = [(4 - len(item)) * '0' + item for item in symbols]
        return symbols

    def _different(self, symbol_1, symbol_2):
        """计算两个码元之间的不同比特数
        """
        cnt = 0
        for bit_1, bit_2 in zip(symbol_1, symbol_2):
            if bit_1 != bit_2:
                cnt += 1
        return cnt

    @property
    def _punish_matrix(self):
        """惩罚矩阵
        表征不同码元之间的不同程度
        """
        matrix = np.zeros((self._num_class, self._num_class), dtype=int)
        for i in range(self._num_class):
            for j in range(self._num_class):
                matrix[i, j] = self._different(self._symbols[i], self._symbols[j])
        return matrix

    def cal_ber(self):
        """计算误比特率
        """
        num_err = np.sum(self._punish_matrix * self._confusion_matrix)
        ber = num_err / (self._num_class * self._num_one_class * 4)
        return ber


if __name__ == '__main__':

    ber = BER(1)
    print(ber._symbols)
    punish_matrix = ber._punish_matrix
    print(punish_matrix * punish_matrix)