# 朴素贝叶斯算法 用 贝叶斯估计 方法估计参数
# 贝叶斯估计， λ=1  K=2， S=3； λ=1 拉普拉斯平滑
# Ni表示第 i个属性可能的取值数 对于触感--只有硬滑和软粘两个属性 xxx

import pandas as pd
import numpy as np


class NavieBayesB(object):
    def __init__(self):
        self.A = 1    # 即 λ=1
        self.K = 2    # 类别有2类：好瓜、坏瓜
        self.S = 3    # 属性值有3类

    def getTrainSet(self):
        trainSet = pd.read_csv('melon.csv')
        # 由dataframe类型转换为数组类型 17x7
        trainSetNP = np.array(trainSet)
        # 获取训练数据 x1,x2,...,x6
        trainData = trainSetNP[:, 0:trainSetNP.shape[1]-1]
        # 训练数据所对应的所属类型Y
        labels = trainSetNP[:, trainSetNP.shape[1]-1]
        return trainData, labels

    def classify(self, trainData, labels, features):
        # 转换为 list 类型
        labels = list(labels)
        # 求先验概率
        P_y = {}
        for label in labels:
            P_y[label] = (labels.count(label) + self.A) / float(len(labels) + self.K*self.A)
        print(P_y)    # {'是':0.474, '否':0.526} {key:value}格式
        # 求条件概率
        P = {}
        for y in P_y.keys():
            y_index = [i for i, label in enumerate(labels) if label == y]   # y在labels中的所有下标
            y_count = labels.count(y)     # y在labels中出现的次数
            for j in range(len(features)):
                p_key = str(features[j]) + '|' + str(y)
                print(p_key)
                x_index = [i for i, x in enumerate(trainData[:,j]) if x == features[j]]   # x在trainData[:,j]中的所有下标
                xy_count = len(set(x_index) & set(y_index))   # x y同时出现的次数
                P[p_key] = (xy_count + self.A) / float(y_count + self.S*self.A)   # 条件概率
                print(P[p_key])
        # features所属类
        F = {}
        for y in P_y.keys():
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y] * P[str(x)+'|'+str(y)]
        # 概率最大值对应的类别
        features_y = max(F, key=F.get)
        return features_y


if __name__ == '__main__':
    nb = NavieBayesB()
    # 训练数据
    trainData, labels = nb.getTrainSet()
    # x1,x2,...,x6
    features = ['青绿', '蜷缩', '清脆', '清晰', '凹陷', '硬滑']
    # 该特征应属于哪一类
    result = nb.classify(trainData, labels, features)
    print(features, '属于', result)
