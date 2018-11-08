# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov 6 2018
SVM Regression for Ali_Zhengqi
"""

__author__ = 'Shida Zheng'

from MLshida.libsvm321.python import svmutil, svm
from MLshida import mlutil as mlu
import numpy as np


class SVMParameters(object):
    """
    SVM相关参数配置类,不同的SVM在这里进行模型调整
    创建对象时导入训练集和标签列,并转换回普通list对象
    """
    def __init__(self, xTrainingIn, yTrainingIn):
        """
        构造函数
        :param xTrainingIn: 训练特征集(numpy.array)
        :param yTrainingIn: 标签(numpy.array)
        """
        self.CScale = [1, 3, 5, 7, 9]  # 惩罚系数C
        self.gammaScale = [-11, -9, -7, -5]  # 高斯核参数gamma
        self.epsilonScale = [-5, -3, -1]  # e-SVM容忍度
        self.cmdBasic = '-s 3 -t 2 -v 5 -h 0'  # libsvm训练参数
        self.xTraining = xTrainingIn
        self.yTraining = yTrainingIn

        self.minMSE = float('inf')
        self.minCIndex = 0
        self.minGammaIndex = 0
        self.minEpsilonIndex = 0

        self.model = svm.svm_model()

    def largerScaleSearching(self):
        C = list(map(lambda x: pow(2, x), self.CScale))
        gamma = list(map(lambda x: pow(2, x), self.gammaScale))
        epsilon = list(map(lambda x: pow(2, x), self.epsilonScale))
        for i in range(len(C)):
            for j in range(len(gamma)):
                for k in range(len(epsilon)):
                    cmd = self.cmdBasic + ' -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k])
                    currMSE = svmutil.svm_train(self.yTraining, self.xTraining, cmd)
                    if currMSE < self.minMSE:
                        self.minMSE = currMSE
                        self.minCIndex = i
                        self.minGammaIndex = j
                        self.minEpsilonIndex = k
                        print('==============================' +
                              'imin=' + str(self.minCIndex) + ',jmin=' + str(self.minGammaIndex) +
                              ',kmin=' + str(self.minEpsilonIndex) + ' --> minMSE = ' + str(self.minMSE) +
                              '================================')
                    print('>>>>>>Stage 1>>>>>>>' + 'i=' + str(i) + ',j=' + str(j) + ',k=' + str(k) +
                          ' --> currMSE = ' + str(currMSE))

    def minorScaleSearching(self, n=10):
        minCScale = 0.5 * (self.CScale[max(1, self.minCIndex - 1)] + self.CScale[self.minCIndex])
        maxCScale = 0.5 * (self.CScale[min(len(self.CScale), self.minCIndex + 1)] + self.CScale[self.minCIndex])
        newCScale = np.arange(minCScale, maxCScale, (maxCScale - minCScale)/n).tolist()

        minGammaScale = 0.5 * (self.gammaScale[max(1, self.minGammaIndex - 1)] + self.gammaScale[self.minGammaIndex])
        maxGammaScale = 0.5 * (self.gammaScale[min(len(self.gammaScale), self.minGammaIndex + 1)]
                               + self.gammaScale[self.minGammaIndex])
        newGammaScale = np.arange(minGammaScale, maxGammaScale, (maxGammaScale - minGammaScale)/n).tolist()

        minEpsilonScale = 0.5 * (self.epsilonScale[max(1, self.minEpsilonIndex - 1)]
                                 + self.epsilonScale[self.minEpsilonIndex])
        maxEpsilonScale = 0.5 * (self.epsilonScale[min(len(self.epsilonScale), self.minEpsilonIndex + 1)]
                                 + self.epsilonScale[self.minEpsilonIndex])
        newEpsilonScale = np.arange(minEpsilonScale, maxEpsilonScale, (maxEpsilonScale - minEpsilonScale)/n).tolist()

        newC = list(map(lambda x: pow(2, x), newCScale))
        newGamma = list(map(lambda x: pow(2, x), newGammaScale))
        newEpsilon = list(map(lambda x: pow(2, x), newEpsilonScale))

        # 默认的最优参数
        minC = newC[int(n/2)]
        minGamma = newGamma[int(n/2)]
        minEpsilon = newEpsilon[int(n/2)]

        for i in range(len(newC)):
            for j in range(len(newGamma)):
                for k in range(len(newEpsilon)):
                    cmd = self.cmdBasic + \
                          ' -c ' + str(newC[i]) + ' -g ' + str(newGamma[j]) + ' -p ' + str(newEpsilon[k])
                    currMSE = svmutil.svm_train(self.yTraining, self.xTraining, cmd)
                    if currMSE < self.minMSE:
                        self.minMSE = currMSE
                        minC = newC[i]
                        minGamma = newGamma[j]
                        minEpsilon = newEpsilon[k]
                        print('==============================' +
                              'minC=' + str(minC) + ',minGamma=' + str(minGamma) + ',minEpsilon=' + str(minEpsilon) +
                              ' --> minMSE = ' + str(self.minMSE) +
                              '================================')
                    print('>>>>>>Stage 2>>>>>>>'
                          + 'i=' + str(i) + ',j=' + str(j) + ',k=' + str(k) + ' --> currMSE = ' + str(currMSE))
        return [minC, minGamma, minEpsilon]

    def getOptimalModel(self, optimalParameters):
        optC = optimalParameters[0]
        optG = optimalParameters[1]
        optE = optimalParameters[2]
        cmd = '-' + self.cmdBasic.strip(' -v 5 -h 0') + ' -c ' + str(optC) + ' -g ' + str(optG) + ' -p ' + str(optE)
        self.model = svmutil.svm_train(yTraining, xTraining, cmd)
        svmutil.svm_save_model('autosave.model', self.model)
        return self.model

    def testForAccuracy(self):
        pass

    def predict(self, testdata, outputPath, loadModelFile=False):
        if not loadModelFile:
            model = self.model
        else:
            model = svmutil.svm_load_model('autosave.model')
            # model = mlu.unpickleSth('/home/shida/ali/Nov/181107_svm_r.model')

        yPred, _, _ = svmutil.svm_predict(list(range(len(testdata))), testdata, model)
        predRes = rNormalization(yPred, meanList[len(self.xTraining)], stdList[len(self.xTraining)])  # 预测结果反归一化
        # predPath = '/home/shida/ali/Nov/predictMetaData.txt'
        # mlu.pickleSth(predPath, predRes)
        mlu.writeResult(outputPath, predRes)
        return None


def splitDataLabel(dataset):
    """
    将特征集和标签分离
    :param dataset:输入最后一列为标签列的数据集
    :return: 分离的特征集和标签列
    """
    features = []
    labels = []
    for line in dataset:
        features.append(line[:-1])
        labels.append(line[-1])
    return features, labels


def rearrangeDataLabel(dataset):
    res = []
    for line in dataset:
        tp = line[-1]
        line[1:] = line[:-1]
        line[0] = tp
        res.append(line)
    return res


def normalization(dataset):
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)

    dataRes = np.zeros(np.shape(dataset))
    for i in range(len(dataset)):
        dataRes[i] = (dataset[i]-mean)/std
    return dataRes, mean, std


def rNormalization(dataset, mean, std):
    dataRes = np.zeros(np.shape(dataset))
    for i in range(len(dataset)):
        dataRes[i] = dataset[i]*std+mean
    return dataRes


if __name__ == '__main__':
    trainData, _, trainTitle = mlu.loadDataSet('/home/shida/ali/Nov/zhengqi_train.txt', dataNum=2888)
    trainData = np.array(trainData)  # 转换为numpy.array格式
    dataNormed, meanList, stdList = normalization(trainData)  # 归一化
    np.random.shuffle(dataNormed)  # 打乱数据

    dataNormedList = dataNormed.tolist()  # 将numpy.array格式转换为list, LIBSVM只接受list格式数据
    xTraining, yTraining = splitDataLabel(dataNormedList)  # 分开数据和标签
    svmModel = SVMParameters(xTraining, yTraining)  # 将训练数据和标签输入SVM参数类中,得到svm训练实例对象

    # svmModel.largerScaleSearching()  # 大尺度上优化参数
    svmModel.minMSE = 0.10369939138206832
    # svmModel.minCIndex = 3
    # svmModel.minGammaIndex = 4
    # svmModel.minEpsilonIndex = 1
    # optParam = svmModel.minorScaleSearching(n=5)  # 小尺度上优化参数
    optParam = [21.112126572366314, 0.0029603839189656167, 0.1435872943746294]
    optModel = svmModel.getOptimalModel(optParam)  # 使用优化的参数训练模型(默认保存为autosave.model)

    # modelPath = '/home/shida/ali/Nov/181107_svm_r.model'
    # mlu.pickleSth(optModel, modelPath)  # 使用pickle保存模型 !!!ctypes objects containing pointers cannot be pickled

    testData, _, testTitle = mlu.loadDataSet('/home/shida/ali/Nov/zhengqi_test.txt', dataNum=2888)
    testData = np.array(testData)
    testNormed, _, _ = normalization(testData)
    testNormedList = list(testNormed.tolist())

    resPath = '/home/shida/ali/Nov/zhengqi_res_svm_20181106-1.txt'
    svmModel.predict(testNormedList, resPath)  # 使用模型预测,并将结果输出到resPath

# Done and Done...
