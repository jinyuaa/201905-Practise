# 数据集取自《周志华-机器学习》西瓜数据
# 首先对数据集进行属性的标注 详细数据信息参考 data1.txt
# 计算得出第一个划分最优属性为纹理(信息增益最大)
# 第二个划分最优属性为根蒂或脐部或触感(三者信息增益同为最大)
# 上述结果与书中所述相同

"""
色泽：0-青绿 1-乌黑 2-浅白
根蒂：0-蜷缩 1-稍蜷 2-硬挺
敲声：0-浊响 1-沉闷 2-清脆
纹理：0-清晰 1-稍糊 2-模糊
脐部：0-凹陷 1-稍凹 2-平坦
触感：0-硬滑 1-软黏
好瓜: no-代表否 yes-代表是
"""

from math import log
import operator
import json


def create_dataset():
    # 数据集
    dataset = [[0, 0, 0, 0, 0, 0, 'yes'],
               [1, 0, 1, 0, 0, 0, 'yes'],
               [1, 0, 0, 0, 0, 0, 'yes'],
               [0, 0, 1, 0, 0, 0, 'yes'],
               [2, 0, 0, 0, 0, 0, 'yes'],
               [0, 1, 0, 0, 1, 1, 'yes'],
               [1, 1, 0, 1, 1, 1, 'yes'],
               [1, 1, 0, 0, 1, 0, 'yes'],
               [1, 1, 1, 1, 1, 0, 'no'],
               [0, 2, 2, 0, 2, 1, 'no'],
               [2, 2, 2, 2, 2, 0, 'no'],
               [2, 0, 0, 2, 2, 1, 'no'],
               [0, 1, 0, 1, 0, 0, 'no'],
               [2, 1, 1, 1, 0, 0, 'no'],
               [1, 1, 0, 0, 1, 1, 'no'],
               [2, 0, 0, 2, 2, 0, 'no'],
               [0, 0, 1, 1, 1, 0, 'no']]
    # 分类属性
    label = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    return dataset, label


# 计算数据集根节点的信息熵
def calculate_comentropy(dataset):
    # dataset 列表长度为 17 返回数据集的行数
    number = len(dataset)
    # print("len(dataset):", number)
    # 保存每个标签（label）出现次数的字典
    labelcounts = {}
    # 对每组特征向量进行统计 统计标签的频次
    for featvex in dataset:
        # 提取标签信息
        current_label = featvex[-1]
        # 如果标签没有放入统计次数的字典，添加进去
        if current_label not in labelcounts.keys():
            labelcounts[current_label] = 0
        # label 计数
        labelcounts[current_label] += 1
        # print("labelcounts:", labelcounts)
    comentropy = 0.0
    # 计算信息熵
    for key in labelcounts:
        # 选择该标签的概率
        prob = float(labelcounts[key]) / number
        # print("key:", labelcounts[key])
        # print("prob:", prob)
        comentropy -= prob * log(prob, 2)
    return comentropy


# 计算信息增益
def choos_feature(dataset):
    # 获取特征个数
    number_label = len(dataset[0]) - 1
    # 计算根节点的信息熵
    root_comentropy = calculate_comentropy(dataset)
    # 信息增益
    best_info_gain = 0.0
    # 最优索引值
    beat_label = -1
    # 遍历所有的特征
    for i in range(number_label):
        # 获得 dataset 的第i个所有特征
        feat_list = [example[i] for example in dataset]
        # 创建set集合{} 元素不可重复
        uniqu_vals = set(feat_list)
        # 初始值操作
        new_comentropy = 0.0
        # 计算信息增益
        for value in uniqu_vals:
            # sun_dataset 划分后的子集 按照每个属性 0-1-2 一次排列的子集
            sub_dataset = split_dataset(dataset, i, value)
            # print("i是{},value是{},sub_dataset是{}".format(i, value, sub_dataset))
            # 计算子集的概率
            prob = len(sub_dataset) / float(len(dataset))
            # print("prob:", prob)
            # 根据公式计算每个子集属性 分支节点的信息熵
            # print("Ent(D%d)为%.3f:" % (value, calculate_comentropy(sub_dataset)))
            new_comentropy += prob * calculate_comentropy(sub_dataset)
        # 计算属性的信息增益
        info_gain = root_comentropy - new_comentropy
        # 打印每个特征的信息增益
        # print("第%d个特征的信息增益为%.8f" % (i, info_gain))
        if info_gain > best_info_gain:
            # 更新信息增益，找到最大的
            best_info_gain = info_gain
            # 记录信息增益最大的索引值
            beat_label = i
    # 返回信息增益最大特征索引
    return beat_label


# 按给定特征划分数据集
# dataset:待划分数据集;axis:划分数据集的特征;value:需要返回的特征值
def split_dataset(dataset, axis, value):
    rest_dataset=[]
    # 按 dataset 列表中的第 axis 列值等于 value 分数据集
    for featvec in dataset:
        # 值等于 value 的每一行为新的列表(去除了第axis个数据)
        if featvec[axis] == value:
            # 删除这一维特征
            reduced_featvec = featvec[:axis]
            reduced_featvec.extend(featvec[axis+1:])
            rest_dataset.append(reduced_featvec)
    return rest_dataset


# 通过排序 返回出现次数最多的类别
def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 递归构建决策树
def create_tree(dataset, labels):
    # 类别向量
    class_list = [example[-1] for example in dataset]
    # 如果只有一个类别，返回
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果所有特征都被遍历完了，返回出现次数最多的类别
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    # 按照信息增益最高选择最优划分属性，索引
    best_feat = choos_feature(dataset)
    # 该最优划分属性的label
    best_feat_label = labels[best_feat]
    # 构建树的字典
    my_tree = {best_feat_label: {}}
    # 已经选择的特征不再参与分类 从 label 的 list 中删除该 label
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    # 该属性所有可能取值，也就是节点的分支
    unique_value = set(feat_values)
    # 对每个分支，递归构建树
    for value in unique_value:
        sub_labels = labels[:]
        # 构建数据的子集合，并进行递归
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree


if __name__ == '__main__':
    dataset, label = create_dataset()
    # print(dataset[0])
    print(label)
    # print(type(dataset))
    print("根节点的信息熵为：", calculate_comentropy(dataset))
    a = choos_feature(dataset)
    print("第一个最优划分特征为：{}".format(label[a]))
    del label[a]
    print("-----------------------------------------------------")
    # 计算第二个最优划分属性
    data1 = split_dataset(dataset, a, 0)
    print("第一个节点的信息熵为：", calculate_comentropy(data1))
    b = choos_feature(data1)
    print("第二个最优划分特征为：{}".format(label[b]))
    del label[b]
    print("-----------------------------------------------------")
    # 计算第三个最优划分属性
    data2 = split_dataset(data1, b, 1)
    print("第二个节点的信息熵为：", calculate_comentropy(data2))
    c = choos_feature(data2)
    print("第三个最优划分特征为：{}".format(label[c]))
    del label[c]
    print("-----------------------------------------------------")
    # 计算第四个最优划分属性
    data3 = split_dataset(data2, c, 1)
    print("第三个节点的信息熵为：", calculate_comentropy(data3))
    d = choos_feature(data3)
    print("第四个最优划分特征为：{}".format(label[d]))
    print("-----------------------------------------------------")

    # 纹理：清晰，根蒂：稍蜷 色泽 没有浅白的 故决策树上没有这个节点 tree为一个字典
    dataset, label = create_dataset()
    tree = create_tree(dataset, label)
    print("决策树字典：")
    print(json.dumps(tree, ensure_ascii=False))

