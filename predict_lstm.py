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
Date: 2019-03-08 
"""

import os
import argparse
import logging
import json
from numpy import argmax
from sklearn.metrics import accuracy_score, confusion_matrix
from FeatureEngineering import DataLoader
from classifier_cnn1d import CNNModel
from config import WORK_DIR
from config import NUM_CLASSES, NUM_EACH_CLASS, LABEL_SELECTED


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float)
    parser.add_argument('--bias', type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # params
    args = parse()
    noise = args.noise
    bias = args.bias
    pred_data_path = 'dataset/test_noise_{:04d}_bias_{:03d}.json'.format(int(1000 * noise), bias)
    train_data_path = 'dataset/noise_{:04d}_bias_{:03d}.json'.format(int(1000 * noise), bias)
    model_dir = 'model/lstm/noise_{:04d}_bias_{:03d}'.format(int(1000 * noise), bias)
    model_name = sorted(os.listdir(model_dir))[-1]
    model_path = os.path.join(model_dir, model_name)


    data = DataLoader(file_path=pred_data_path, num_classes=NUM_CLASSES, num_each_class=1000,
                      discrete=True, labels=LABEL_SELECTED, test_size=0.)
    feature_predict = data.run_predict(discrete=True)

    # 加载模型
    model = CNNModel(data.data, feature_predict=feature_predict)
    y_pred = model.run_predict(model_path=model_path)
    y_true = argmax(data.data['Y_total'], axis=1)


    results = {}
    acc = accuracy_score(y_true, y_pred)
    c_matrix = confusion_matrix(y_true, y_pred).tolist()
    results['acc'] = acc
    results['confusion_matrix'] = c_matrix
    with open('evaluation/lstm/noise_{:04d}_bias_{:03d}.json'.format(int(1000 * noise), bias), 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)