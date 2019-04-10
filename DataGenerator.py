#!/usr/bin/python
# -*- coding:utf-8 -*-
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The Common Tools Project
"""
基于机器学习的OAM低截获通信系统设计 -- 数据生成模块

Usage: python DataGenerator.py mixed_modes_selected.txt --bias 0 --K 15 --noise 0.02
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019-03-26
"""

import argparse
import json
import math
import logging
import numpy as np
from utils.log import init_log

# config
NUM_SAMPLING = 10000
RAD_SAMPLING = np.arange(-math.pi, math.pi, 2 * math.pi / NUM_SAMPLING).reshape(-1, 1)[:NUM_SAMPLING]
NUM_ONE_CLASS_TRAIN = 100
NUM_ONE_CLASS_TEST = 1000


def _get_phase(p_complex):
    """计算相位
    需根据象限判断取值，[-180, 180]
    """
    real = p_complex.real
    imag = p_complex.imag

    if abs(real) < 1e-5 and abs(imag) < 1e-5:
        return 0
    elif abs(real) < 1e-5 and imag > 0:
        return 90
    elif abs(real) < 1e-5 and imag < 0:
        return -90

    if real > 0 and imag > 0:
        return math.degrees(math.atan(abs(imag) / abs(real)))
    elif real > 0 and imag < 0:
        return -math.degrees(math.atan(abs(imag) / abs(real)))
    elif real < 0 and imag > 0:
        return 180 - math.degrees(math.atan(abs(imag) / abs(real)))
    else:
        return -180 + math.degrees(math.atan(abs(imag) / abs(real)))


class OAMSignal(object):
    """单模态OAM类

    Attribute:
        _mode: int, OAM模态
        _amp: float, 幅度, 默认为1
        _phase_center: float, 相位, 单位rad, 默认为0
    """
    def __init__(self, mode, amp=1, phase_center=0):
        self._mode = mode
        self._amp = np.ones((NUM_SAMPLED, 1)) * amp
        self._phase = RAD_SAMPLED * self._mode + phase_center

    def seq(self):
        """获取周向复数序列，包含幅相信息
        """
        return self._amp * np.exp(1j * self._phase)

    def seq_sampled(self, n_samples, deg_min=-35, deg_max=35, deg_bias=0):
        """仅获取采样数据

        Args:
            n_samples: 采样点数
            deg_min: 最小采样角度
            deg_max: 最大采样角度
        """
        portion_min = (deg_min + deg_bias - (-180)) / (180 - (-180))
        portion_max = (deg_max + deg_bias - (-180)) / (180 - (-180))
        start = int(NUM_SAMPLED * portion_min)
        end = int(NUM_SAMPLED * portion_max)
        step = int(NUM_SAMPLED * (portion_max - portion_min) / (n_samples - 1))
        amp_sampled = self._amp[start:end + int(0.5 * step):step]
        phase_sampled = self._phase[start:end + int(0.5 * step):step]

        return amp_sampled * np.exp(1j * phase_sampled)


class Mixer(object):
    """模态叠加类

    Attributes:
        _amp_array: numpy.array, 归一化幅度系数
        _modes_array: numpy.array, 模态组合
        _noise_factor: float, 噪声强度系数
    """
    def __init__(self, amp_array, mode_array, noise_factor, n_samples, bias):
        assert amp_array.shape == mode_array.shape
        self._amp_array = amp_array / amp_array.sum()
        self._mode_array = mode_array
        self._noise_factor = noise_factor
        self._n_samples = n_samples
        self._bias = bias
        self._mixed_sampled = None
        self._mixed_noised_sampled = None
        self.intensity = None
        self.phase = None

    def _mix(self):
        """完整混合序列
        """
        mixed = np.zeros((NUM_SAMPLED, 1), dtype=np.complex64)
        for amp, mode in np.hstack((self._amp_array, self._mode_array)):
            oam = OAMSignal(mode=mode, amp=amp)
            mixed += oam.seq()
        return mixed

    def _mix_sampled(self):
        """采样混合序列
        """
        mixed_sampled = np.zeros((self._n_samples, 1), dtype=np.complex64)
        for amp, mode in np.hstack((self._amp_array, self._mode_array)):
            oam = OAMSignal(mode=mode, amp=amp)
            mixed_sampled += oam.seq_sampled(deg_bias=self._bias, n_samples=self._n_samples)
        return mixed_sampled

    def _add_noise(self, seq):
        """添加噪声
        """
        dim1, dim2 = seq.shape
        noise = self._noise_factor * (np.random.randn(dim1, dim2) + np.random.randn(dim1, dim2) * 1j)
        seq_noised = seq + noise
        return seq_noised

    def _to_dB(self, seq):
        """转换成分贝
        """
        intensity = 20 * np.log10(np.abs(seq))
        phase = np.apply_along_axis(func1d=_get_phase, axis=1, arr=seq)
        return intensity, phase

    def generate_for_dataset(self):
        """采样并混合
        用于生成数据集，先采样可以降低运算成本
        Step 1: 采样，混合
        Step 2: 添加噪声
        Step 3: 转换分贝
        """
        mixed_sampled = self._mix_sampled()
        noised = self._add_noise(mixed_sampled)
        intensity, phase = self._to_dB(noised)
        return intensity, phase

    def generate_for_plot(self):
        """仅混合
        用于生成绘图数据，不进行采样操作
        """
        mixed = self._mix()
        noised = self._add_noise(mixed)
        intensity, phase = self._to_dB(noised)
        return intensity, phase

    def cal_snr(self):
        """计算SNR
        """
        mixed = self._mix()
        noised = self._add_noise(mixed)
        snr = 10 * np.log10(np.sum(np.abs(pow(mixed, 2))) / (np.sum(np.abs(pow(noised, 2))) - np.sum(np.abs(pow(mixed, 2)))))
        return snr


class DataGenerator(object):
    """数据产生类
    用于生成指定规模和参数的数据集
    """
    def __init__(self, filename, num_one_class, noise_factor, n_sampled, bias):
        self._filename = filename
        self._num_one_class = num_one_class
        self._noise_factor = noise_factor
        self._n_sampled = n_sampled
        self._bias = bias
        self.data = {
            'X': list(),
            'y': list()
        }

    def run(self):
        """主接口
        生成训练数据
        """
        with open(self._filename, 'r') as f:

            # 获取数据配置
            for i, line in enumerate(f):
                amps = [int(amp) for amp in line.split(',')[0].split(':')[-1].strip().split(' ')]
                modes = [int(mode) for mode in line.split(',')[1].split(':')[-1].strip().split(' ')]
                amp_array = np.array(amps).reshape(-1, 1)
                mode_array = np.array(modes).reshape(-1, 1)

                # 生成每一个数据实例
                for j in range(self._num_one_class):
                    mixer = Mixer(amp_array=amp_array, mode_array=mode_array, noise_factor=self._noise_factor,
                                  n_samples=self._n_sampled, bias=self._bias)
                    intensity, phase = mixer.generate_for_dataset()
                    self.data['X'].append(intensity.reshape(-1,).tolist() + phase.reshape(-1,).tolist())
                    self.data['y'].append(i)

                print('Data of label {} generated.'.format(i))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate dataset. '
                                     'Usage: python DataGenerator.py mixed_modes_selected.txt --bias 0 --K 15 --noise 0.02')
    parser.add_argument('mode_filename')
    parser.add_argument('--noise', type=float)
    parser.add_argument('--bias', type=int)
    parser.add_argument('--K', type=int)
    args = parser.parse_args()
    mode_filename = args.mode_filename
    noise = args.noise
    bias = args.bias
    K = args.K


    # logger
    init_log('./logs/data_gen_K_{:02d}_noise_{:04d}_bias_{:03d}'.format(K, int(noise * 1000), bias))


    # generate
    logging.info('Generating train data of K_{:02d}_noise_{:04d}_bias_{:03d}'.format(K, int(noise * 1000), bias))
    data_train = DataGenerator(filename=mode_filename, num_one_class=NUM_ONE_CLASS_TRAIN,
                               noise_factor=noise, n_sampled=K, bias=bias)
    data_train.run()
    logging.info('Generating test data of K_{:02d}_noise_{:04d}_bias_{:03d}'.format(K, int(noise * 1000), bias))
    data_test = DataGenerator(filename=mode_filename, num_one_class=NUM_ONE_CLASS_TEST,
                              noise_factor=noise, n_sampled=K, bias=bias)
    data_test.run()


    # save
    save_path_train = 'dataset/K_{:02d}/noise_{:04d}_bias_{:03d}.json'.format(K, int(noise * 1000), bias)
    save_path_test = 'dataset/K_{:02d}/test_noise_{:04d}_bias_{:03d}.json'.format(K, int(noise * 1000), bias)
    with open(save_path_train, 'w') as f:
        json.dump(data_train.data, f, indent=4)
    with open(save_path_test, 'w') as f:
        json.dump(data_test.data, f, indent=4)
