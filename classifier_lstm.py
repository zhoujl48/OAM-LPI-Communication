#!/usr/bin/python
# -*- coding:utf-8 -*-
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The Common Tools Project
"""
基于机器学习的OAM低截获通信系统设计 -- LSTM序列模型

Usage: python classifier_lstm.py --data_path data/data_noise_004_bias_0.json
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019-03-04 
"""
import os
import argparse
import logging
from datetime import datetime
from numpy import argmax
from keras import regularizers, Model
from keras.models import load_model
from keras.layers import Input, Embedding, LSTM, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from utils import log
from FeatureEngineering import DataLoader
from config import WORK_DIR, SIZE_ONE_SAMPLE, NUM_EACH_CLASS, NUM_CLASSES, LABEL_SELECTED


class LSTMModel(object):
    """LSTM序列模型

    Attributes:
        data:
            _feature_train: 训练数据
            _label_train: 训练标签
            _feature_test: 测试数据
            _feature_label: 测试标签
        layers:
            _dense_size_first: Dense第一层大小
            _dense_size_middle: Dense除头尾层之外的中间层的大小列表
            _dense_size_last: Dense最后一层大小
        params:
            _regular: 正则化系数
            _epoch: 训练轮数
            _batch_size: 批大小
            _dropout_size: dropout系数
            _save_path_base: 模型保存路径
            _tb_log_dir: TensorBoard的日志路径
    """

    def __init__(self, data_split, save_path_base=None, epoch=100, batch_size=128, dropout_size=0.5, tb_log_dir=None, regular=0.001,
                 input_len=30, embedding_size=128, discrete_num=24000, lstm_size=64, dense_size=128, output_size=16, feature_predict=None):

        # data
        self._feature_train = data_split['X_train']
        self._label_train = data_split['Y_train']
        self._feature_test = data_split['X_test']
        self._label_test = data_split['Y_test']
        self._feature_predict = feature_predict

        # layers
        self._input_len = input_len
        self._embedding_size = embedding_size
        self._discrete_num = discrete_num
        self._lstm_size = lstm_size
        self._dense_size = dense_size
        self._output_size = output_size

        # params
        self._regular = regular
        self._epoch = epoch
        self._batch_size = batch_size
        self._dropout_size = dropout_size
        self._save_path_base = save_path_base
        self._tb_log_dir = tb_log_dir

    def model(self):
        """建模训练
        Step 1: 定义模型结构
        Step 2: 确定优化方式
        Step 3: 定义回调函数（模型保存）
        Step 4: 模型拟合
        """

        # 定义模型结构
        feature_input = Input(shape=(self._input_len,))
        feature_embedding = Embedding(output_dim=self._embedding_size, input_dim=self._discrete_num + 1,
                                      input_length=self._input_len)(feature_input)
        feature_embedding_dropout = Dropout(self._dropout_size)(feature_embedding)
        input_lstm = LSTM(units=self._lstm_size, return_sequences=True)(feature_embedding_dropout)
        dropout_lstm = Dropout(self._dropout_size)(input_lstm)
        flatten_lstm = Flatten()(dropout_lstm)
        dense = Dense(self._dense_size, activation='relu', kernel_regularizer=regularizers.l1(self._regular))(flatten_lstm)
        dropout_dense = Dropout(self._dropout_size)(dense)
        output = Dense(self._output_size, activation='softmax')(dropout_dense)
        model = Model([feature_input], [output])
        model.summary()


        # 优化目标
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Callbacks -- 模型保存
        checkpoint = ModelCheckpoint(self._save_path_base + '.{epoch:03d}-{val_acc:.4f}.hdf5',
                                     monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        # Callbacks -- TensorBoard
        if self._tb_log_dir != '':
            tensorboard = TensorBoard(log_dir=self._tb_log_dir, histogram_freq=0, batch_size=32,
                        write_graph=True, write_grads=False, write_images=False, update_freq='epoch',
                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
            callbacks_list = [checkpoint, tensorboard]
        else:
            callbacks_list = [checkpoint]

        # 模型训练
        model.fit(self._feature_train,
                  self._label_train,
                  epochs=self._epoch,
                  batch_size=self._batch_size,
                  callbacks=callbacks_list,
                  validation_data=(self._feature_test, self._label_test))

    def run(self):
        """离线训练接口
        """
        self.model()

    def run_predict(self, model_path):
        """离线预测接口
        """
        model = load_model(filepath=model_path, compile=False)
        Y_pred = model.predict(self._feature_predict)
        y_pred = argmax(Y_pred, axis=1)


        return y_pred
if __name__ == '__main__':

    # parameters
    parser = argparse.ArgumentParser(''
                                     'Usage: python classifier_mlp.py --data_path data/data_noise_004_bias_0.json')
    parser.add_argument('--data_path', help='set data source', type=str)
    parser.add_argument('--epoch', help='set the training epochs', default=100, type=int)
    parser.add_argument('--batch_size', help='set the training batch size', default=128, type=int)
    parser.add_argument('--regular', help='set the training regularization', default=0.0001, type=float)
    parser.add_argument('--dropout_size', help='set the training dropout', default=0.5, type=float)
    parser.add_argument('--tb', help='\'True\' means using TensorBoard, \'False\' means not using TensorBoard', type=bool, default=False)
    parser.add_argument('--embedding_size', help='set the embedding size', type=int, default=64)
    parser.add_argument('--lstm_size', help='set the lstm layer size', type=int, default=64)
    parser.add_argument('--dense_size', help='set the dense layer size', type=int, default=64)
    args = parser.parse_args()
    data_path = args.data_path
    data_type = data_path.split('/')[-1].split('.')[0]
    epoch = args.epoch
    batch_size = args.batch_size
    regular = args.regular
    dropout_size = args.dropout_size
    tb = args.tb
    embedding_size = args.embedding_size
    lstm_size = args.lstm_size
    dense_size = args.dense_size

    # 日志
    log.init_log(os.path.join(WORK_DIR, 'logs', data_type))

    # 导入数据
    logging.info('Loading data: {}'.format(data_path))
    data = DataLoader(file_path=data_path, num_classes=NUM_CLASSES, num_each_class=NUM_EACH_CLASS,
                      discrete=True, labels=LABEL_SELECTED)
    data.run()

    # 模型保存路径
    model_dir = os.path.join(WORK_DIR, 'model', 'lstm', data_type)
    if not os.path.exists(model_dir):
        logging.info('Makedir: {}'.format(model_dir))
        os.makedirs(model_dir)
    save_dir_base = os.path.join(model_dir, 'lstm_embedding_{embedding_size}_lstm_{lstm_size}_batch_{batch_size}_regular_{regular}_dropout_{dropout}'\
                                 .format(embedding_size=embedding_size, lstm_size=lstm_size, batch_size=batch_size,
                                         regular=regular, dropout=dropout_size))

    # TensorBoard日志路径
    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    tb_log_dir = os.path.join(WORK_DIR, 'logs', 'tb_{data_type}_{model}_{stamp}'.format(data_type=data_type,
                                                                                        model=save_dir_base.split('/')[-1],
                                                                                        stamp=time_stamp))

    # 模型训练
    model = LSTMModel(data_split=data.data, save_path_base=save_dir_base,tb_log_dir=tb_log_dir,
                      regular=regular, epoch=epoch, batch_size=batch_size, dropout_size=dropout_size,
                      embedding_size=embedding_size, lstm_size=lstm_size, dense_size=dense_size)
    model.run()


