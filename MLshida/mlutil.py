# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov 6 2018
Some useful utilities for machine learning
"""

__author__ = 'Shida Zheng'

import pickle


def loadDataSet(filename, dataNum=2200, hasTitle=True):
    """
    导入本地数据
    :param filename: 文件全路径
    :param dataNum: 训练样本与验证样本切分点
    :param hasTitle: 本地数据是否含表头
    :return: 训练数据, 验证数据, 表头
    """
    with open(filename) as fr:
        datamat = []
        dataeval = []
        title = []
        reader = fr.readlines()  # list
        if hasTitle:
            title = reader[0]
        for line in reader:
            if hasTitle and line == title:
                continue
            elif len(datamat) < dataNum:
                cutLine = line.strip().split('\t')
                floatLine = list(map(float, cutLine))
                datamat.append(floatLine)
            else:
                cutLine = line.strip().split('\t')
                floatLine = list(map(float, cutLine))
                dataeval.append(floatLine)

    return datamat, dataeval, title


def writeResult(filename, result):
    """
    预测结果输出至文件
    :param filename: 文件全路径
    :param result: 预测结果
    :return: None
    """
    with open(filename, 'w') as fo:
        print(result)
        for out in result:
            fo.writelines(str(out) + '\n')

    return None


def pickleSth(sth, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(sth, fp)
    return None


def unpickleSth(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)
