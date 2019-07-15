#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataSet (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: helen
"""

import os
import numpy as np
import operator


def file2matrix(file_name):
    """
    将文本文件中记录解析为Numpy数据
    :param file_name: 文本文件
    :return: 样本数据和类别数据
    """
    fr = open(file_name)
    lines = fr.readlines()
    rows = len(lines)
    data_mat = np.zeros((rows, 3))
    class_label = []
    index = 0
    for line in lines:
        list_line = line.strip().split('\t')   # [特征1，特征2，特征3，label]
        data_mat[index, :] = list_line[0:3]
        class_label.append(int(list_line[-1]))
        index += 1

    return data_mat, class_label


def auto_norm(data_set):
    """
    数据归一化：数字较大的属性对结果产生较大影响，因此处理不同取值范围的特征值，将属性值归一化为0~1之间
    new_val = (old_val - min) / (max - min)
    :param data_set:
    :return:
    """
    min_values = data_set.min(0)  # min(0):列的最小值  min():整个矩阵最小值  min(1):行的最小值
    max_values = data_set.max(0)
    ranges = max_values - min_values

    m = data_set.shape[0]
    norm_mat = data_set - np.tile(min_values, (m, 1))  # np.tile(data. (m,n)): 把data扩展为行的m倍,列的n倍
    norm_mat = norm_mat / np.tile(ranges, (m, 1))      # 矩阵元素相除

    # # -------第二种实现方式---start---------------------------------------
    # norm_mat = (data_set - min_values) / ranges
    # # -------第二种实现方式---end---------------------------------------------

    return norm_mat, ranges, min_values


def classify(inX, data_set, labels, k):
    """
    K近邻算法实现
    伪代码：1、计算已知类别数据集中的每个点与新样本之间的距离
           2、按照距离递增排序
           3、选择与当前距离最小的K个点
           4、确定前k的点所在类别的频率
           5、返回前k个点频率最高的列别作为当前点的预测分类
    :param inX: 新样本
    :param data_set: 样本数据集
    :param labels: label集
    :param k: 超参数
    :return: 新样本所属类别
    """
    size = data_set.shape[0]  # 样本数
    """
    欧氏距离：点到点之间的距离
        第一行： 同一个点 到 dataSet 的第一个点的距离。
        第二行： 同一个点 到 dataSet 的第二个点的距离。
        ...
        第N行： 同一个点 到 dataSet 的第N个点的距离。
    """
    diff_mat = np.tile(inX, (size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)  # axis=1:行向量相加
    sq_distance = np.sqrt(sq_distance)

    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y
    sorted_distinct = sq_distance.argsort()

    # 统计前k个最近样本对应的类别数
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distinct[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # 根据频率字段,降序排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]   # sorted_class_count：[(label,count),(label,count)]


def date_class_test():
    """
    对约会网站的测试方法，并将分类错误的数量和分类错误率打印出来
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    ho_ratio = 0.10
    data_mat, data_label = file2matrix('datingTestSet2.txt')
    norm_mat, _, _ = auto_norm(data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)  # 设置测试样本数
    error_count = 0.0
    for i in range(num_test_vecs):
        result = classify(norm_mat[i, :], norm_mat[num_test_vecs:m, :], data_label[num_test_vecs:m], 5)
        print("the classifier back with:%d, the real is:%d" % (result, data_label[i]))
        if result != data_label[i]:
            error_count += 1
    print("the total error:%d, and total vecs:%d" % (error_count, num_test_vecs))


def classify_person():
    """  预测新样本 """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percent ot time spent playing games?"))
    fly_miles = float(input("fly miles per year?"))
    ice_cream = float(input("ice cream consumed per year?"))

    data_mat, data_label = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(data_mat)
    in_arr = np.array([fly_miles, percent_tats, ice_cream])

    result = classify(in_arr-min_vals / ranges, norm_mat, data_label, 3)
    print("You will probably like this person: ", result_list[result-1])


def img2vector(file_name):
    """
    将图像转换为向量格式
    输入数据的图片格式是 32 * 32的
    该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    return_vect = np.zeros((1, 1024))
    with open(file_name) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                return_vect[0, 32*i+j] = int(line[j])

    return return_vect


def hand_writing_class_test():
    """ 手写数字识别 """
    hw_labels = []
    train_file_list = os.listdir('trainingDigits')
    m = len(train_file_list)
    training_mat = np.zeros((m, 1024))

    # 训练集合样本特征的label
    for i in range(m):
        file_name_str = train_file_list[i]
        file_str = file_name_str.split('.')[0]
        # 训练样本label
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        # 训练样本特征
        training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)

    # 测试集合样本特征和label
    test_file_list = os.listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)

        classifier_result = classify(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with:%d, the real answer is %d" % (classifier_result, class_num))
        if classifier_result != class_num:
            error_count += 1
        print("the total number of errors is: %d" % error_count)
        print("the total error rate is: %d" % (error_count / float(m_test)))


if __name__ == "__main__":
    date_class_test()

    classify_person()

    hand_writing_class_test()
