#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
线性回归：预测目标值是连续值
"""
import numpy as np
import matplotlib.pyplot as plt


class Regression:
    """线性回归类"""
    def load_data_set(self, filename):
        num_feat = len(open(filename).readline().split('\t')) - 1  # 特征数
        data_mat, label_mat = [], []
        fr = open(filename)
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
        return data_mat, label_mat

    def stand_regres(self, x_arr, y_arr):
        """最佳拟合参数"""
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr).T
        xTx = x_mat.T * x_mat
        if np.linalg.det(xTx) == 0.0:  # 行列式为0，没有逆矩阵
                print("This matrix is singular, cannot do inverse")
                return
        ws = xTx.I * (x_mat.T * y_mat)  # 拟合参数公式
        return ws

    def predict(self, ws, x_arr):
        x_mat = np.mat(x_arr)
        y_hat = ws * x_mat  # 预测值
        return y_hat

    def draw_line(self, x_arr, y_arr, ws):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr).T

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_mat[:, 1].flatten().A[0], y_mat[:, 0].flatten().A[0])  # 散列点

        x_mat.sort()
        y_hat = x_mat * ws
        ax.plot(x_mat[:, 1], y_hat)  # 拟合直线
        plt.show()

    def corrcoef(self, x_arr, y_arr, ws):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr)
        y_hat = x_mat * ws

        # corrcoef相关系数是用以反映变量之间相关关系密切程度的统计指标,必须是行向量
        return np.corrcoef(y_mat, y_hat.T)


class LocalWeightLineRegression:
    """局部加权线性回归"""
    def lwlr(self, test_point, x_arr, y_arr, k):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr).T
        m = np.shape(x_mat)[0]  # 样本数
        weights = np.mat(np.eye((m)))
        # 计算权重
        for j in range(m):
            diff_mat = test_point - x_mat[j, :]  # 第j个样本与新样本的距离
            weights[j, j] = np.exp(diff_mat * diff_mat.T) / (-2*k**2)  # 权重公式

        xTx = x_mat.T * (weights * x_mat)
        if np.linalg.det(xTx) == 0.0:
            return

        ws = xTx.I * (x_mat.T * (weights * y_mat))  # 拟合参数公式
        return test_point * ws

    def lwlr_test(self, test_arr, x_arr, y_arr, k):
        m = np.shape(test_arr)[0]
        y_hat = np.zeros(m)
        for i in range(m):
            y_hat[i] = self.lwlr(test_arr[i], x_arr, y_arr, k)
        return y_hat

    def drwa_lwlr(self, x_arr, y_arr, y_hat, k):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr)
        srt_ind = x_mat[:, 1].argsort(0)  # 根据第1列元素排序
        x_sort = x_mat[srt_ind][:, 0, :]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_sort[:, 1], y_hat[srt_ind])
        ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T.flatten().A[0], s=2, c='red')

        plt.title("k=%f" % k)
        plt.show()


class Ridge:
    """
    岭回归：数据缩减方法，解决特征比样本多的情况（特征比样本多无法求X.T*Xde逆矩阵）
    回归系数：w = （X.T*X+ alpha*I）.I * X.T*Y
    最初用于处理特征数大于样本数的情况，现在也用于在估计中加入偏差，通过引入惩罚项减少不重要的参数
    """

if __name__ == "__main__":
    lr = Regression()
    x, y = lr.load_data_set('ex0.txt')
    # ws = lr.stand_regres(x, y)
    # lr.draw_line(x, y, ws)
    # print(lr.corrcoef(x, y, ws))

    lwlr = LocalWeightLineRegression()
    y_hat = lwlr.lwlr_test(x, x, y, 0.0001)
    # TODO：不同k应该是不同拟合直线，不知道什么原因这个没有生效
    lwlr.drwa_lwlr(x, y, y_hat, 0.0001)

