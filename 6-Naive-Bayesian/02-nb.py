# 极大似然估计的方法估计参数
# 数据集为 机器学习-周志华 西瓜数据 实验结果与书中一致
# 某个属性值在训练集中没有与某个类同时出现过，会导致计算概率为 0 的情况

import pandas as pd
import numpy as np


class NaiveBayes(object):
    def getTrainSet(self):
        dataSet = pd.read_csv('melon.csv')
        # print(dataSet)
        # 将数据由 dataframe 类型转换为数组类型 17x7
        dataSetNP = np.array(dataSet)
        print(dataSetNP)
        # 获取训练数据 x1,x2,...,x6
        trainData = dataSetNP[:, 0:dataSetNP.shape[1]-1]
        # 获取训练数据所对应的所属类型 Y
        labels = dataSetNP[:, dataSetNP.shape[1]-1]
        return trainData, labels

    # 求 labels 中每个label的先验概率
    def classify(self, trainData, labels, features):
        # 转换为 list类型
        labels = list(labels)
        # 存入label的概率 字典类型
        P_y = {}
        for label in labels:
            # p = count(y) / count(Y)
            P_y[label] = labels.count(label)/float(len(labels))
        print(P_y)    # {'是':0.471, '否':0.529} {key:value}格式
        # 求 label与 feature 同时发生的概率
        P_xy = {}
        for y in P_y.keys():
            # labels中出现y值的所有数值的下标索引
            y_index = [i for i, label in enumerate(labels) if label == y]
            # features[0] 在 trainData[:,0]中出现的值的所有下标索引
            for j in range(len(features)):
                x_index = [i for i, feature in enumerate(trainData[:,j]) if feature == features[j]]
                # set(x_index)&set(y_index)列出两个表相同的元素
                xy_count = len(set(x_index) & set(y_index))
                p_key = str(features[j]) + '*' + str(y)
                P_xy[p_key] = xy_count / float(len(labels))

        #求条件概率
        P = {}
        for y in P_y.keys():
            for x in features:
                p_key = str(x) + '|' + str(y)
                print(p_key)
                P[p_key] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])    # P[X1|Y] = P[X1,Y]/P[Y]
                print(P[p_key])
        # 求 ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']所属类别
        F = {}   # ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']属于各个类别的概率
        for y in P_y:
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y]*P[str(x)+'|'+str(y)]     # P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
        print(F)
        features_label = max(F, key=F.get)  # 概率最大值对应的类别
        return features_label


if __name__ == '__main__':
    nb = NaiveBayes()
    # 训练数据
    trainData, labels = nb.getTrainSet()
    # x1,x2
    features = ['青绿', '蜷缩', '清脆', '清晰', '凹陷', '硬滑']
    # 该特征应属于哪一类
    result = nb.classify(trainData, labels, features)
    print(features, '属于', result)
