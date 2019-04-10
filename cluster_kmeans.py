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
Date: 2019-03-26 
"""
import os
import json
import argparse
import logging
from random import sample, seed
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from BER import BER
from utils.log import init_log


# 数据集参数
PARAMS_DATASET = {
    'noise': [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050],
    'bias': [0]
}
# 数据集参数
# PARAMS_DATASET = {
#     'noise': [0.030],
#     'bias': list(range(0, 60, 10))
# }


def load_data(filename):
    """加载数据集
    """
    with open(filename) as f:
        data = json.load(f)
        X = data['X']
        y = data['y']
    return X, y


def save_results(results, noise, bias, K, num_each_class=1000):
    """结果保存JSON
    每组数据集参数，保存一个结果文件
    """
    save_name = 'noise_{:04d}_bias_{:03d}.json'.format(int(noise * 1000), bias)
    save_path = os.path.join('evaluation', 'K_{:02d}'.format(K), 'kmeans', save_name)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def sample_data(X, num_each_class=1000, total_num_each_class=1000, num_class=16):

    X_sampled = []
    for i in range(num_class):
        seed(0)
        idx_sample = sample(list(range(i * total_num_each_class, (i + 1) * total_num_each_class)), num_each_class)
        X_sampled_one_class = [X[idx] for idx in idx_sample]
        X_sampled.extend(X_sampled_one_class)

    return X_sampled


def keams_match(X_test, n_clusters, num_each_class=1000, num_class=16):

    results = {}

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_test)

    # 按聚落大小排序
    Y_pred = np.zeros((n_clusters, n_clusters), dtype=int)
    for i, label in enumerate(kmeans.labels_):
        cluster_true = i // num_each_class
        Y_pred[cluster_true, label] += 1
    Y_pred_sorted = np.array(sorted(Y_pred.tolist(), key=lambda x: x.index(max(x))), dtype=int)

    # 计算准确率
    true_pred_cnt = Y_pred_sorted.max(axis=0)
    results['acc'] = sum(true_pred_cnt) / (num_each_class * num_class)

    return results






if __name__ == '__main__':

    # logger
    init_log('logs/svm_search')

    # 参数设置
    parser = argparse.ArgumentParser(''
                                     'Usage: python classifier_svm.py --K 7')
    parser.add_argument('--K', type=int)
    parser.add_argument('--num_each_class', type=int, default=1000)
    args = parser.parse_args()
    K = args.K
    num_each_class = args.num_each_class

    acc_list = []
    for noise in PARAMS_DATASET['noise']:
        for bias in PARAMS_DATASET['bias']:

            # 加载数据
            logging.info('Loading dataset of noise({}) and bias({})...'.format(noise, bias))
            data_test = 'dataset/K_{:02d}/test_noise_{:04d}_bias_{:03d}.json'.format(K, int(noise * 1000), bias)
            X_test, y_test = load_data(data_test)

            # 采样
            X_test_sampled = sample_data(X_test, num_each_class)

            # # 标准化
            # scaler = StandardScaler().fit(X_test_sampled)
            # X_scaled_test = scaler.transform(X_test_sampled)

            # 聚类并计算准确率
            logging.info('Clustering...'.format(noise, bias))
            results = keams_match(X_test_sampled, n_clusters=16, num_each_class=num_each_class)

            acc_list.append(results['acc'])

            # # 保存结果
            # save_results(results, noise, bias, K=K)


    print(acc_list)