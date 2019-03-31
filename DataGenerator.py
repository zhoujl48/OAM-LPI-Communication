#!/usr/bin/python
# -*- coding:utf-8 -*-
#
# Copyright (c) 2019 ***.com, Inc. All Rights Reserved
# The Common Tools Project
"""
基于机器学习的OAM低截获通信系统设计 -- 数据生成模块

Usage: python DataGenerator.py mixed_modes.txt dataset/noise_002_bias_000.json 0.02
Authors: Zhou Jialiang
Email: zjl_sempre@163.com
Date: 2019-03-26
"""

import argparse
import json
import numpy as np

# config
NUM_SAMPLED = 10000
_2PI_RAD = 2 * np.pi
RAD_SAMPLED = np.arange(-_2PI_RAD / 2, _2PI_RAD / 2, _2PI_RAD / NUM_SAMPLED).reshape(-1, 1)[:NUM_SAMPLED]
_2PI_DEG = 360
DEG_SAMPLED = np.arange(-_2PI_DEG / 2, _2PI_DEG / 2, _2PI_DEG / NUM_SAMPLED).reshape(-1, 1)[:NUM_SAMPLED]
RAD_TO_DEG = _2PI_DEG / _2PI_RAD
DEG_TO_RAD = _2PI_RAD / _2PI_DEG


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

    def seq_sampled(self, n_samples=15, deg_min=-35, deg_max=35, deg_bias=0):
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
        amp_sampled = self._amp[start:end:step]
        phase_sampled = self._phase[start:end:step]

        return amp_sampled * np.exp(1j * phase_sampled)


class Mixer(object):
    """模态叠加类

    Attributes:
        _amp_array: numpy.array, 归一化幅度系数
        _modes_array: numpy.array, 模态组合
        _noise_factor: float, 噪声强度系数
    """
    def __init__(self, amp_array, mode_array, noise_factor, n_samples=15, bias=0):
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
            mixed_sampled += oam.seq_sampled(deg_bias=self._bias)
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
        phase = np.arctan(seq.imag / seq.real) * RAD_TO_DEG
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
    def __init__(self, filename, num_one_class, noise_factor, n_sampled=15):
        self._filename = filename
        self._num_one_class = num_one_class
        self._noise_factor = noise_factor
        self._n_sampled = n_sampled
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
                    mixer = Mixer(amp_array=amp_array, mode_array=mode_array, noise_factor=self._noise_factor)
                    intensity, phase = mixer.generate_for_dataset()
                    self.data['X'].append(intensity.reshape(-1,).tolist() + phase.reshape(-1,).tolist())
                    self.data['y'].append(i)

                print('Data of label {} generated.'.format(i))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate dataset. '
                                     'Usage: python DataGenerator.py mixed_modes.txt dataset/noise_002_bias_000.json 0.02')
    parser.add_argument('mode_filename')
    parser.add_argument('save_filename')
    parser.add_argument('noise_factor', help='parameter of noise', type=float)
    parser.add_argument('--num_sampled', type=int, default=15)
    parser.add_argument('--bias', type=int, default=0)
    parser.add_argument('--num_one_class', type=int, default=100)
    args = parser.parse_args()
    mode_filename = args.mode_filename
    save_filename = args.save_filename
    noise_factor = args.noise_factor
    num_sampled = args.num_sampled
    bias = args.bias
    num_one_class = args.num_one_class


    data_gen = DataGenerator(filename=mode_filename, num_one_class=num_one_class, noise_factor=noise_factor)
    data_gen.run()

    with open(save_filename, 'w') as f:
        json.dump(data_gen.data, f, indent=4)
