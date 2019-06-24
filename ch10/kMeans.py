#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
K-means算法是一种聚类算法，属于无监督学习，原理：用户随机指定k个簇新，样本点根据到簇心的距离判断属于哪个簇，簇的质心为该簇下所有样本的坐标均值迭代更新，最后可以得到k个簇
    可知：簇内样本越相似越好，簇与簇越远越好
    优点：简单，容易理解    缺点：由于所有样本点都要计算距离，所以速度慢
伪代码：
    创建k个点作为起始质心
    当任意一个点的簇分配结果发生改变时：
        对数据集中的每个数据点：
            对每个质心：
                计算质心到数据点之间的距离（欧式距离）
            将数据点分配到距离最近的簇
        对每一个簇，计算簇中所有点的均值，并将均值作为质心
"""
import numpy as np


class KMeans:
    def load_data(self, file_name):
        data_mat = []
        with open(file_name) as fr:
            for line in fr.readlines():
                cur_line = line.strip().split('\t')
                flt_line = list(map(float, cur_line))  # map:python2中返回一个list，python3中返回的是一个map对象
                data_mat.append(flt_line)
        return data_mat

    def dist_eclud(self, vec_a, vec_b):
        # 计算两个向量的距离，使用欧式距离
        return np.sqrt(np.sum(np.power(vec_a - vec_b), 2))

    def rand_cent(self, data_set, k):
        """
        初始化k个随机簇
        :param data_set: 训练集
        :param k: 超参数，用户指定k个簇
        :return:
        """
        data_mat = np.mat(data_set)
        n = np.shape(data_mat)[1]  # 特征数,也就是维度
        centroids = np.mat(np.zeros((k, n)))  # 簇心矩阵：KXN
        for j in range(n):
            minJ = min(data_mat[:, j])  # 在训练集中找第j个维度的最小值
            rangeJ = float(max(data_mat[:, j]) - minJ)  # 第j个维度的最大值-最小值
            centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
        return centroids

    def k_means(self, data_set, k, dist_meas=dist_eclud, create_cent=rand_cent):
        """
        K-均值聚类算法
        :param data_set: 训练数据集
        :param k: 簇心数
        :param dist_meas: 距离计算公式，默认为欧式距离
        :param create_cent: 初始化质心函数，默认为随机初始化
        :return:
        """


if __name__ == "__main__":
    kmeans = KMeans()
    x = kmeans.load_data('testSet.txt')
    cent = kmeans.rand_cent(x, 2)
    print(cent)


