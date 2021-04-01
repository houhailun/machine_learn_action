#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
kNN: k Nearest Neighbors
@Time    : 2019/7/15 15:22
@Author  : helen
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap


class KNN:
    def __init__(self, neighbor=3):
        iris = datasets.load_iris()
        self.X = iris.data[:, :2]
        self.y = iris.target
        self.neighbors = neighbor
        self.h = .02  # 网格中的步长

    def model(self):
        # 创建彩色的图
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        for weights in ['uniform', 'distance']:
            # 参数说明:
            # n_neighbors: 用来确定多数投票规则的邻居数K
            # weights : 在进行分类判断的时候给最近邻的点加上权重，它的默认值是'uniform',也就是等权重，
            #   所以在这种情况下我们就可以使用多数投票规则来判断输入实例的类别预测。
            #   还有一个选择是'distance',是按照距离的倒数给定权重
            # algorithm 是分类时采取的算法，有 {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}，默认为auto，会自动进行选择最合适的算法
            # p: p=1，距离方法定义为曼哈顿距离，在p=2的时候我们定为欧几里得距离。默认值为2
            clf = neighbors.KNeighborsClassifier(n_neighbors=self.neighbors, weights=weights)
            clf.fit(self.X, self.y)

            # 绘制决策边界。为此，我们将为每个分配一个颜色来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
            x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
            y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, self.h),
                                 np.arange(y_min, y_max, self.h))  # 绘制网格
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # 将结果放入一个彩色图中
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # 绘制训练点
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap_bold)
            plt.xlim(xx.min(), xx.max())  # 刻度
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (self.neighbors, weights))
            plt.show()


if __name__ == "__main__":
    knn = KNN()
    knn.model()
