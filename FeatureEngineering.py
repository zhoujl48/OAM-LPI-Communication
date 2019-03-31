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
Date: 2019-03-04 
"""
import os
import json
import numpy as np
import logging
from collections import defaultdict
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.log import init_log
from config import LABEL_SELECTED


class DataLoader(object):
    """数据加载类
    
    Attributes:

        Data Config:
            _file_path: str, 数据地址
            _num_classes: int, 样本类别数
            _num_each_class: int, 每一类样本数

        Data Split:
            _labels: list, 指定要加载的数据标签列表
            _test_size: float, 分割数据集时的测试集占比
            _random_state: int, 分割数据集时的随机数种子

        Data Discretization:
            _decimals: int, 数据离散化的精度, 即保留原连续变量的小数位

        Flags:
            _discrete: bool, 设置是否离散化, 默认False
            _scale: bool, 设置是否进行标准化, 默认True

        Data:
            data: 用于预测的完整数据集
            data: 划分完毕的数据集
            data: 离散化的数据集
            scaler: 标准化转换器实例
            data: 最终可以馈入模型的数据集
    """
    def __init__(self, file_path, num_classes, num_each_class, labels=None, test_size=0.2, random_state=1,
                 discrete=False, decimals=1, scale=True):
        assert os.path.exists(file_path)
        
        # config
        self._file_path = file_path
        self._num_classes = num_classes
        self._num_each_class = num_each_class
        self._test_size = test_size
        self._random_state = random_state
        self._labels = labels

        # discretization
        self._discrete = discrete
        self._decimals = decimals

        # normalization
        self._scale = scale
        self.scaler = None
        
        # data
        self.data = {
            'X_total': None,
            'Y_total': None,
            'X_train': None,
            'Y_train': None,
            'X_test': None,
            'Y_test': None,
        }

    def _label_selection(self, X, y):
        """类别选择
        根据labels选择指定标签，根据标签加载对应的数据

        Args:
             X: 原始数据集的特征矩阵
             Y: 原始数据集的标签矩阵

        Return:
            X_selected: 标签选择后的特征矩阵
            y_selected: 标签选择后的标签矩阵
        """

        if self._labels is None:
            return X, y

        X_selected = np.zeros(shape=(0, X.shape[1]))
        y_selected = np.zeros(shape=(0,))
        flag_label_used = defaultdict(bool)
        label_new = -1
        for i in range(y.shape[0]):
            if y[i] in self._labels:
                if not flag_label_used[y[i]]:
                    label_new += 1
                    flag_label_used[y[i]] = True
                X_selected = np.append(X_selected, X[i, :].reshape(1, -1), axis=0)
                y_selected = np.append(y_selected, label_new)
        print(y_selected.min(), y_selected.max())
        assert X_selected.shape[0] == y_selected.shape[0]
        logging.info('Shape of X_selected and y_selected: {}, {}'.format(X_selected.shape, y_selected.shape))

        return X_selected, y_selected

    def load(self):
        """加载数据
        Step 1: 加载原始数据
        Step 2: 根据指定labels进行选择
        Step 3: 将标签矩阵转化为one-hot形式
        """
        with open(self._file_path) as f:
            data = json.load(f)
            X = np.array(data['X']).reshape((self._num_classes * self._num_each_class, -1))
            y = np.array(data['y']).reshape(self._num_classes * self._num_each_class,)
        X_selected, y_selected = self._label_selection(X, y)
        Y_selected = to_categorical(y_selected, num_classes=len(LABEL_SELECTED))

        self.data['X_total'] = X_selected
        self.data['Y_total'] = Y_selected


    def split(self):
        """加载数据
        Step 1: 读取JSON文件数据
        Step 2: 将标签转化为one_hot模式
        Step 3: 分割训练集与测试集
        """
        # split
        X_train, X_test, Y_train, Y_test = train_test_split(self.data['X_total'], self.data['Y_total'],
                                                            test_size=self._test_size, random_state=self._random_state)

        # data
        self.data['X_train'] = X_train
        self.data['Y_train'] = Y_train
        self.data['X_test'] = X_test
        self.data['Y_test'] = Y_test


    def _discretize(self, X, pwr_min=-60, pwr_max=0, deg_min=-90, deg_max=90):
        """特征离散化
        针对一个特征数据矩阵(X_train或X_test)，
        分别对其功率和相位数据执行:
            Step 1: clip, 去除过大或过小值
            Step 2: 离散化, 保留一位小数并扩大十倍
            Step 3: 添加bias, 使得离散后的数据映射到非负整数集，且须确保功率和相位取值区间无交集
        最后横向合并功率和相位矩阵，完成离散化处理
    
        Args:
            X: 待离散化的特征矩阵
            decimals: 离散精细度，即保留原连续变量的小数位
            pwr_min: 执行clip操作时, 功率最小值, (单位/dB)
            pwr_max: 执行clip操作时, 功率最大值, (单位/dB)
            deg_min: 执行clip操作时, 相位最小值, (单位/度)
            deg_max: 执行clip操作时, 相位最大值, (单位/度)
    
        Return:
            X_discrete: 离散化处理后的特征数据矩阵
        """
    
        assert len(X.shape) == 2
    
        # 功率离散化
        pwr = X[:, :15]
        pwr_clipped = np.clip(pwr, pwr_min, pwr_max)
        pwr_discrete = np.array(np.around(pwr_clipped, decimals=self._decimals) * pow(10, self._decimals), dtype=int)
        pwr_bias = (pwr_max - pwr_min) * pow(10, self._decimals)
        pwr_featured = pwr_discrete + pwr_bias
    
        # 相位离散化
        deg = X[:, 15:]
        deg_clipped = np.clip(deg, deg_min, deg_max)
        deg_discrete = np.array(np.around(deg_clipped, decimals=self._decimals) * pow(10, self._decimals), dtype=int)
        deg_bias = (0 - deg_min + pwr_max - pwr_min) * pow(10, self._decimals)
        deg_featured = deg_discrete + deg_bias
    
        # 合并特征矩阵
        X_discrete = np.hstack((pwr_featured, deg_featured))

        return X_discrete

    def discretization(self):
        """数据集离散化
        将数据集中的X_train和X_test均进行离散化

        Args:
            decimals: 离散精细度，即保留原连续变量的小数位
        """
        assert self.data is not None

        # 离散化
        X_train_discrete = self._discretize(self.data['X_train'])
        X_test_discrete = self._discretize(self.data['X_test'])

        # 特征离散化
        self.data['X_train'] = X_train_discrete
        self.data['X_test'] = X_test_discrete


    def normalization(self):
        """标准化
        标准化特征矩阵，并保存标准化系数(均值/方差)
        """
        assert self.data is not None

        self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(self.data['X_train'])

        X_train_scaled = self.scaler.transform(self.data['X_train'])
        X_test_scaled = self.scaler.transform(self.data['X_test']) \
            if self.data['X_test'].shape[0] > 0 else self.data['X_test']

        self.data['X_train'] = X_train_scaled
        self.data['X_test'] = X_test_scaled


    def run(self):
        """主接口 -- 用于模型训练
        """

        # 若需离散化
        if self._discrete:
            self.load()
            self.split()
            self.discretization()

        # 否则进行标准化
        else:
            self.load()
            self.split()
            self.normalization()

    def run_predict(self, discrete=False):
        """主接口 -- 用于模型训练
        """
        if discrete:
            self.load()
            return self._discretize(self.data['X_total'])
        else:
            self.load()
            return self.data['X_total']






# 用于调试特征工程模块
if __name__ == '__main__':

    # logger
    init_log('./logs/FeatureEngineering')

    # config
    data_path = './data/data_noise_002_bias_0.json'
    num_classes = 23
    num_each_class = 100
    test_size = 0.2
    random_state = 1

    # 实例化数据加载类
    data = DataLoader(file_path=data_path, num_classes=num_classes, num_each_class=num_each_class, labels=LABEL_SELECTED)

    # 加载，标签选取，并划分
    data.run()
    print(data.data['X_train'][0])
    print(data.data['Y_train'], data.data['Y_train'].shape)
    print(data.scaler.mean_, data.scaler.var_)

    # # 加载，标签选取，划分，并离散化
    # data_discrete = DataLoader(file_path=data_path, num_classes=num_classes, num_each_class=num_each_class,
    #                            discrete=True, labels=LABEL_SELECTED)
    # data_discrete.run()
    # print(data_discrete.data['X_train'][0])
    # print(data_discrete.data['X_train'].max(),
    #       data_discrete.data['X_train'].min())


