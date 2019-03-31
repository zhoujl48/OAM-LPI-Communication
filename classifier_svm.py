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
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from BER import BER
from utils.log import init_log


# 待优化参数
PARAMS_OPTIMIZE = [
    # {
    #     'kernel': ['rbf'],
    #     'C': [0.0001, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 100, 200],
    #     'gamma': [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    # },
    {
        'kernel': ['linear'],
        'C': [0.0001, 0.001, 0.1, 0.2, 0.5, 1, 5, 10],
    }
]

# 数据集参数
PARAMS_DATASET = {
    'noise': [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050],
    'bias': [0]
}


def load_data(filename):
    """加载数据集
    """
    with open(filename) as f:
        data = json.load(f)
        X = data['X']
        y = data['y']
    return X, y


def save_results(results, noise, bias):
    """结果保存JSON
    每组数据集参数，保存一个结果文件
    """
    save_name = 'noise_{:04d}_bias_{:03d}.json'.format(int(noise * 1000), bias)
    save_path = os.path.join('evaluation', 'svm', save_name)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def best_model_and_results(X_train, y_train, X_test, y_test):
    """训练、评估
    网格搜索法遍历优化参数

    Return:
        results: 结果字典，包含
            best_params - 最佳模型参数
            best_params - 最佳模型的K折验证准确率
            pred_acc - 预测准确率
            confusion_matrix - 预测混淆矩阵
            ber - 预测误比特率
    """

    results = {}

    # train and search
    svc = svm.SVC()
    clf = GridSearchCV(svc, PARAMS_OPTIMIZE, n_jobs=-1, scoring='accuracy', cv=10)
    clf.fit(X=X_train, y=y_train)

    # predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    c_matrix = confusion_matrix(y_test, y_pred)
    ber = BER(confusion_matrix=c_matrix, num_one_class=1000).cal_ber()

    results['best_params'] = clf.best_params_
    results['best_score'] = clf.best_score_
    results['pred_acc'] = acc
    results['confusion_matrix'] = c_matrix.tolist()
    results['ber'] = ber

    return results


if __name__ == '__main__':

    # logger
    init_log('logs/svm_search')

    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='search on single dataset or the total dataset, \'single\' or \'total\'', default='total')
    parser.add_argument('--noise', type=float)
    parser.add_argument('--bias', type=int)
    args = parser.parse_args()
    mode = args.mode


    # 若指定数据集参数，则只对该数据集训练模型
    if args.mode is 'single':

        # 加载数据
        logging.info('Loading dataset of noise({}) and bias({})...'.format(args.noise, args.bias))
        data_train = 'dataset/noise_{:04d}_bias_{:03d}.json'.format(int(args.noise * 1000), args.bias)
        data_test = 'dataset/test_noise_{:04d}_bias_{:03d}.json'.format(int(args.noise * 1000), args.bias)
        X_train, y_train = load_data(data_train)
        X_test, y_test = load_data(data_test)

        # 网格搜索，模型评估
        logging.info('Searching best SVM...'.format(args.noise, args.bias))
        results = best_model_and_results(X_train, y_train, X_test, y_test)
        print(results)


    # 遍历所有数据集
    else:
        for noise in PARAMS_DATASET['noise']:
            for bias in PARAMS_DATASET['bias']:

                # 加载数据
                logging.info('Loading dataset of noise({}) and bias({})...'.format(noise, bias))
                data_train = 'dataset/noise_{:04d}_bias_{:03d}.json'.format(int(noise * 1000), bias)
                data_test = 'dataset/test_noise_{:04d}_bias_{:03d}.json'.format(int(noise * 1000), bias)
                X_train, y_train = load_data(data_train)
                X_test, y_test = load_data(data_test)

                # 网格搜索，效果评估
                logging.info('Searching best SVM for dataset of noise({}) and bias({})...'.format(noise, bias))
                results = best_model_and_results(X_train, y_train, X_test, y_test)
                save_results(results, noise, bias)


