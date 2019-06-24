#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
集成学习：学习多个弱学习器，线性组合成一个强学习器，提升性能
    bagging：通过自助汇聚法，从训练集中随机抽取样本，得到新的样本训练集，学习到弱学习器1，多次后可以学习到多个弱学习器，通过线性组合成一个强学习器，分类任务通过投票法实现组合
    boosting：通过修改样本权值来更新学习若学习器，利用加权线性组合成一个强学习器
    区别：bagging是并行，boosting是串行；bagging不修改样本权值，boosting修改样本权值；bagging的弱学习器没有权值，boosting弱学习器有权值
    boosting算法描述：
        （1）初始化样本权值：D1=(W11,...,W1m), 其中W1i = 1/m
        （2）对于n = 1，2，...，M来说：
            a.使用样本权值D1学习弱分类器: Gm(x):X->{-1, +1}
            b.计算Gm(x)的分类误差率: e = 未正确分类样本数 / 所有样本数
            c.Gm(x)的系数:alpha = 1/2 ln(1-e / e)
            d.更新样本权值
        （3）加权线性组合得到强学习器和最终分类函数
"""
import numpy as np


class AdaBoost:
    def load_simple_data(self):
        dat_mat = np.matrix([[1.0, 2.1],
                             [2.0, 1.1],
                             [1.3, 1.0],
                             [1.0, 1.0],
                             [2.0, 1.0]])
        class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]

        return dat_mat, class_labels

    def stump_classify(self, data_matrix, dimen, thresh_val, thresh_ineq):
        # 小于阈值的数据类别位-1，大于阈值的数据类别位+1
        ret_array = np.ones(np.shape(data_matrix)[0], 1)  # mX1的向量
        if thresh_ineq == 'lt':
            ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
        else:
            ret_array[data_matrix[:, dimen] > thresh_val] = -1.0

    def build_stump(self, data_arr, class_labels, D):
