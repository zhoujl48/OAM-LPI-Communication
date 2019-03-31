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

# Config
WORK_DIR = '/Users/zhoujl/workspace/oam'
SIZE_ONE_SAMPLE = 30
NUM_CLASSES = 16
NUM_EACH_CLASS = 100
LABEL_SELECTED = list(range(16))



DENSE_SIZE_LIST = [256, 64, 64, len(LABEL_SELECTED)]
