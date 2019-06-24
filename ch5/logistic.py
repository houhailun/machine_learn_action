#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
模块说明：逻辑回归模型
算法描述：不同于线性回归的预测值是连续值，LR预测值是离散值，典型的为二分类0，1，这里引用sigmod函数实现
"""

import numpy as np
import matplotlib.pyplot as plt


class LR(object):
    """逻辑回归类实现"""
    def __init__(self):
        pass

    def load_data_set(self):
        fr = open('testSet.txt')
        data_mat = []
        label_mat = []
        for line in fr.readlines():
            line_arr = line.strip().split()
            # X0为1.0
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
        return data_mat, label_mat

    @staticmethod
    def sigmoid(int_x):
        return 1.0 / (1 + np.exp(-int_x))

    def grad_ascent(self, data_mat, label_mat):
        """
        梯度上升优化算法：沿着梯度方向移动总是能达到最大或最小值
        伪代码：
            每个回归系数初始化为1
            重复R次：
                计算整个数据集的梯度
                使用alpha * gradient更新回归系数的向量
                返回回归系数
        """
        _data_mat = np.mat(data_mat)
        _label_mat = np.mat(label_mat).transpose()
        m, n = np.shape(_data_mat)  # m个样本，维度为n
        alpha = 0.001               # 步长
        max_cycles = 500            # 最大迭代次数
        weights = np.ones((n, 1))
        for k in range(max_cycles):
            h = self.sigmoid(_data_mat * weights)
            error = _label_mat - h
            weights = weights + alpha * _data_mat.transpose() * error
        return weights

    def plot_bestfit(self, wei, data_mat, label_mat):
        """绘制数据集和最佳拟合直线"""
        weights = wei.getA()  # 矩阵转换为array
        data_arr = np.array(data_mat)
        n = np.shape(data_arr)[0]  # 样本数
        # xcord1, ycord1代表正例特征
        # xcord2, ycord2代表负例特征
        xcode1 = xcode2 = ycode1 = ycode2 = []
        for i in range(n):
            if int(label_mat[i]) == 1:
                xcode1.append(data_arr[i, 1])  # data_arr[i][1]
                ycode1.append(data_arr[i, 2])
            else:
                xcode2.append(data_arr[i, 1])
                ycode2.append(data_arr[i, 2])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcode1, ycode1, s=30, c='red', marker='s')
        ax.scatter(xcode2, ycode2, s=30, c='green')
        x = np.arange(-3.0, 3.0, 0.1)
        """
        y的由来？
        首先理论上是这个样子的。
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        w0*x0+w1*x1+w2*x2=f(x)
        x0最开始就设置为1， x2就是我们画图的y值。0是两个类别的分界处
        所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
        """
        y = (-weights[0] - weights[1] * x) / weights[2]
        ax.plot(x, y)
        plt.xlabel('x1')
        plt.xlabel('x2')
        plt.show()

    def random_grad_ascent(self):
        """
        随机梯度上升算法，伪代码：
        所有回归系数初始化为1
        对数据集中每个样本：
            计算该样本的梯度
            使用alpha * gradient更新回归系数
        返回回归系数值
        """
        _data_mat = self.data_mat
        _label_mat = self.label_mat
        m, n = np.shape(_data_mat)  # m个样本，维度为n
        alpha = 0.001  # 步长
        weights = np.ones(n)
        for i in range(m):
            h = self.sigmoid(sum(_data_mat[i] * weights))
            error = _label_mat[i] - h
            weights = weights + alpha * _data_mat[i] * error
        return weights


if __name__ == "__main__":
    lr = LR()
    mat, labels = lr.load_data_set()
    wei = lr.grad_ascent(mat, labels)
    #print(wei)
    lr.plot_bestfit(wei, mat, labels)
    #wei = lr.random_grad_ascent()
    print(wei)