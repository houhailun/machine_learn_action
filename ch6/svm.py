#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
模块说明：支持向量机SVM是当前不做修改即可很好的分类器
"""

import numpy as np


class SVM:
    def load_data_set(self, filename):
        data_mat = []
        label_mat = []
        with open(filename) as fr:
            for line in fr.readlines():
                line_arr = line.strip().split('\t')
                data_mat.append([float(line_arr[0]), float(line_arr[1])])
                label_mat.append(float(line_arr[2]))
        return data_mat, label_mat

    def select_j_random(self, i, m):
        """
        随机选择一个不等于i的alpha下标
        :param i: 第一个alpha的下表
        :param m: 所有alpha的数目，也就是样本数目
        :return: 选择的下标，位于（0，m）之间
        """
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def clip_alpha(self, aj, H, L):
        """调整大于H或者小于L的alpha值"""
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def smo_simple(self, data_mat_in, class_labels, C, toler, max_iter):
        """
        简化版SMO算法
        SMO伪代码：
        创建一个alpha向量并初始化为0
        当迭代次数小于最大迭代次数时：（外循环）
            对数据集中的每个数据向量：（内循环）
                如果该数据向量可以被优化：
                    随机选择另外一个数据向量
                    同时优化这两个向量
                    如果两个向量都不能被优化，退出内循环
            如果所有向量都没有被优化，增加迭代次数，继续下次优化
        :param data_mat_in: 训练数据样本集
        :param class_labels: 训练数据label集
        :param C: 惩罚因子
        :param toler: 容错率
        :param max_iter: 最大迭代次数
        :return:
        """
        data_matrix = np.mat(data_mat_in)
        label_matrix = np.mat(class_labels)
        b = 0
        m, n = np.shape(data_matrix)    # m：样本数量 n:样本维度
        alphas = np.mat(np.zeros((m, 1)))  # 设置alpha为m行1列的列向量
        iter = 0
        while iter < max_iter:
            alpha_pairs_changed = 0  # 表示是否优化
            for i in range(m):
                # fxi表示当前样本预测的类别，公式：alpha*label*(X*Xi)+b，注意：multiply和*都是对应元素相乘
                fxi = float(np.multiply(alphas, label_matrix).T * (data_matrix*data_matrix[i, :].T)) + b
                Ei = fxi - float(label_matrix[i])  # 预测误差
                # 如果alpha[i]可以被优化
                if ((label_matrix[i]*Ei < -toler) and (alphas[i] < C)) or ((label_matrix[i]*Ei > toler) and (alphas[i] > 0)):
                    j = self.select_j_random(i, m)
                    fxj =

if __name__ == "__main__":
    svm = SVM()
    data_arr, label_arr = svm.load_data_set('testSet.txt')
    print(data_arr)