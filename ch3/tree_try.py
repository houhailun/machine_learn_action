#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/15 16:18
# Author: Hou hailun

# 决策树，一种基本的分类回归模型，有ID3，C4.5，CART
# 特征选择，树的建立，树的剪枝

import math
import operator


class MyDecisionTree:
    # 决策树类ID3
    def __init__(self):
        self.tree = {}  # 使用字典来保存树

    @staticmethod
    def create_data_set():
        data_set = [[1, 1, 'yes'],
                    [1, 1, 'yes'],
                    [1, 0, 'no'],
                    [0, 1, 'no'],
                    [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']  # 并非目标变量
        return data_set, labels

    @staticmethod
    def major_cnt(class_list):
        # 计算样本集中类别最多的类别
        # class_cnt = {}
        # for vote in class_list:
        #     if vote not in class_cnt:
        #         class_cnt[vote] = 0
        #     class_cnt[vote] += 1
        #
        # class_cnt = class_cnt.sorted(class_cnt.items(), key=operator.getitem(1), reverse=True)
        # return class_cnt[0][0]

        # 或者利用Counter
        import collections
        return collections.Counter(class_list).most_common(1)[0]

    @staticmethod
    def entropy(data_set):
        """
        计算样本集的熵
        H(D) = -sum(pro * log(pro))
        pro = |Ck| / |D|
        :param data_set:
        :return:
        """
        num_entropy = len(data_set)

        # 计算每个类别对应的样本数据
        class_cnt = {}
        for example in data_set:
            vote = example[-1]
            if vote not in class_cnt:
                class_cnt[vote] = 0
            class_cnt[vote] += 1

        # 计算熵
        shannon_entropy = 0
        for vote in class_cnt:
            prob = class_cnt[vote] / num_entropy
            shannon_entropy -= prob / math.log(prob, 2)
        return shannon_entropy

    @staticmethod
    def split_data_set(data_set, feat_ix, value):
        """
        根据特征的取值对样本集进行划分
        :param data_set:
        :param feat_ix: 特征下标
        :param value: 特征的取值
        :return:
        """
        sub_data_set = []
        for example in data_set:
            if example[feat_ix] == value:
                reduced_feat_vec = example[:feat_ix]
                reduced_feat_vec.extend(example[feat_ix+1:])
                sub_data_set = sub_data_set.append(reduced_feat_vec)
        return sub_data_set

    def choose_best_feature(self, data_set):
        """
        选取最优特征，判断规则为信息增益
        :param data_set:
        :return:
        """
        # 计算香农熵
        shannon_entropy = self.entropy(data_set)

        # 计算每个特征每个取值的信息增益
        num_features = len(data_set[0]) - 1  # Features数, 最后一列是label列
        best_info_gain = 0
        best_feature = -1
        for i in range(num_features):
            # 获取第i个特征的所有取值
            feat_list = [example[i] for example in data_set]
            unique_values = set(feat_list)
            new_entropy = 0.0  # 特征i取值x时对应的条件熵
            for value in unique_values:
                # 对特征的取值划分数据集,并计算条件熵
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / len(data_set)
                new_entropy += prob * self.entropy(sub_data_set)

            # 信息增益
            info_gain = shannon_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    def create_tree(self, data_set, labels):
        """
        创建树
        :param data_set: 样本集合
        :param labels: label名称
        :return:
        """
        # label
        class_list = [example[-1] for example in data_set]

        # 第一个停止条件: 所有的样本都属于同一个类别，则把这些样本设置为改类别
        if len(class_list) == len(class_list[0]):
            return class_list[0]

        # 第二个停止条件: 如果数据集只有1列，则遍历完所有特征时，返回出现次数最多的类别
        if len(data_set[0]) == 1:
            return self.major_cnt(class_list)

        # 正常情况：选取最佳特征
        best_feat = self.choose_best_feature(data_set)

        tmp_lables = labels.copy()
        best_feat_label = tmp_lables[best_feat]
        my_tree = {best_feat_label: {}}  # 字典保存
        del tmp_lables[best_feat]  # 删除已选特征

        # 最佳特征的对应取值
        feat_values = [example[best_feat] for example in data_set]
        unique_values = set(feat_values)

        # 根据所选特征的取值把原始样本集划分为若干个小的样本集，并递归构建树
        for value in unique_values:
            sub_labels = tmp_lables[:]  # 求出剩余的标签label
            # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
            my_tree[best_feat_label][value] = self.create_tree(self.split_data_set(data_set, best_feat, value),
                                                               sub_labels)

        self.tree = my_tree
        return self.tree

