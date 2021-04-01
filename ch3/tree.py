#!/usr/bin/env python
# -*- coding:utf-8 -*-
# <AUTHOR>: Helen

"""
模块功能：决策树是用于分类和回归的树模型，具有很好的可解释性，主要分为：特征选择、树的构造、树的剪枝；主要算法ID3，C4.5，CART
"""
import math
import operator
# import decisionTreePlot as dtPlot


class ID3DecisionTree(object):
    """
    该决策树仅仅是决策树的最简版本，其局限性：
        1、不能处理连续取值的属性值，
        2、不能处理某些属性没有取值的数据，
        3、没有剪枝策略，
        4、数据没有预处理机制，
        5、控制参数缺失，如最大树深等，
        6、不支持多变量决策树。
        该类的使用方法：
            ID3 = ID3DecisionTree()
            myTree = ID3.createTree(myDat, labels)  # 输入训练数据和标签，构造决策树
            predict = ID3.classify(myTree, labels, testData)  # 输入构造的树，标签集合，测试数据（只能是一条）
    """
    def __init__(self):
        self.my_tree = {}

    def entropy(self, data_set):
        """ 计算信息熵"""

        # 数据集的总行数
        num_entries = len(data_set)

        # 为所有可能的分类创建字典,label_cnt={'label1':cnt,'label2':cnt,}
        label_cnt = {}
        for feat_vec in data_set:
            curr_label = feat_vec[-1]         # 提取标签信息
            if curr_label not in label_cnt:   # 统计标签对应的样本数
                label_cnt[curr_label] = 0
            label_cnt[curr_label] += 1

        # 计算熵
        shannon_entropy = 0.0
        for key in label_cnt:
            prob = float(label_cnt[key]) / num_entries  # 使用所有类标签的发生频率计算类别出现的概率
            shannon_entropy -= prob * math.log(prob, 2)

        return shannon_entropy

    def split_dataset(self, data_set, axis, value):
        """
        根据特征和特征取值划分训练集
        :param data_set: 训练集
        :param axis: 该特征在数据集中的index，即第x个特征
        :param value: 特征的取值
        :return: 特征值等于value的数据子集
        """
        ret_data_set = []
        # 划分思路：对样本集中的每一条样本，判断第axis特征当前取值是否等于value
        # 不等于则不处理；等于则取其他的数据
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                reduced_feat_vec = feat_vec[0:axis]
                reduced_feat_vec.extend(feat_vec[axis+1:])
                ret_data_set.append(reduced_feat_vec)

        return ret_data_set

    def choose_best_feature(self, data_set):
        """
        选择切分数据集的最佳特征
        :param data_set: 需要切分的数据集
        :return: 最优特征
        """
        num_features = len(data_set[0]) - 1    # Features数, 最后一列是label列
        base_entropy = self.entropy(data_set)  # 数据集的熵
        best_info_gain = 0.0  # 信息增益
        best_feature = -1     # 最佳特征
        for i in range(num_features):
            # 获取第i个特征的所有取值
            feat_list = [example[i] for example in data_set]
            unique_vals = set(feat_list)
            new_entropy = 0.0
            # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
            for value in unique_vals:
                sub_data_set = self.split_dataset(data_set, i, value)
                # 条件熵H(D|i)
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * self.entropy(sub_data_set)
            # 信息增益 = H(D) - H(D|i)
            # 信息增益是熵的减少或者是数据无序度的减少，比较所有特征中的信息增益，返回最好特征划分的索引值。
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    def create_data_set(self):
        """
        创建数据集
        :return:
        """
        data_set = [[1, 1, 'yes'],
                    [1, 1, 'yes'],
                    [1, 0, 'no'],
                    [0, 1, 'no'],
                    [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']  # 并非目标变量
        return data_set, labels

    def majority_cnt(self, class_list):
        """
        选择出现次数最多的一个结果
        :param class_list: label列的集合
        :return: bestFeature 最优的特征列
        """
        class_count = {}
        # 类标签频率
        for vote in class_list:
            if vote not in class_count:
                class_count[vote] = 0
            class_count[vote] += 1
        # 依据value由大到小排序
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

        # # -----------majorityCnt的第二种方式 start------------------------------------
        # major_label = Counter(classList).most_common(1)[0]
        # return major_label
        # # -----------majorityCnt的第二种方式 end------------------------------------

        return sorted_class_count[0][0]

    def create_tree(self, data_set, labels):
        """
        创建决策树
        :param data_set:
        :param labels:
        :return:
        """
        class_list = [example[-1] for example in data_set]  # 类标签list

        # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]

        # 第二个停止条件: 如果数据集只有1列，遍历完所有特征时，返回出现次数最多的类别
        if len(data_set[0]) == 1:
            return self.majority_cnt(class_list)

        tmp_labels = labels.copy()
        best_feat = self.choose_best_feature(data_set)  # 选择最优的列，得到最优列对应的label含义
        best_feat_label = tmp_labels[best_feat]         # 获取label的名称
        my_tree = {best_feat_label: {}}
        del(tmp_labels[best_feat])                                      # 去除最佳特征
        feat_values = [example[best_feat] for example in data_set]      # 获取最佳特征对应的所有属性值
        unique_vals = set(feat_values)
        # 递归
        for value in unique_vals:
            sub_labels = tmp_labels[:]  # 求出剩余的标签label
            # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
            my_tree[best_feat_label][value] = self.create_tree(self.split_dataset(data_set, best_feat, value), sub_labels)

        self.my_tree = my_tree
        return self.my_tree

    def classify(self, input_tree, feat_labels, test_vec):
        """
        决策树分类函数
        :param input_tree:
        :param feat_labels:
        :param test_vec:
        :return:
        """
        first_str = list(input_tree.keys())[0]     # 获取tree的根节点对于的key值
        second_dict = input_tree[first_str]        # 通过key得到根节点对应的value
        feat_index = feat_labels.index(first_str)  # 找到类别标签的索引

        class_label = 0.0
        for key in second_dict.keys():             # 第二层判断
            if test_vec[feat_index] == key:        # 找到使用哪颗子树
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.classify(second_dict[key], feat_labels, test_vec)  # 继续分类
                else:
                    class_label = second_dict[key]  # 树的叶子节点
        return class_label

    def store_tree(self, input_tree, file_name):
        """ 将之前训练好的决策树模型存储起来，使用 pickle 模块 """
        import pickle
        with open(file_name, 'w') as fw:
            pickle.dump(input_tree, fw)

    def grab_tree(self, file_name):
        """ 将之前存储的决策树模型使用 pickle 模块还原出来 """
        import pickle
        with open(file_name) as fr:
            return pickle.load(fr)


if __name__ == "__main__":
    ID3 = ID3DecisionTree()

    myDat, labels = ID3.create_data_set()
    ID3.create_tree(myDat, labels)
    ID3.classify(ID3.my_tree, labels, [1, 0])
