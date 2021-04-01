#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
模块功能：朴素贝叶斯算法
    思想：基于贝叶斯定理和特征条件独立性假设，先统计先验概率和联合概率，然后求最大后验概率
    注意：朴素贝叶斯算法是求样本属于某个实例的概率
    伪代码：
        计算属于每个类别的概率 P(y=Ck)
        对每篇训练文档：
            对每个类别：
                如果词条出现在文档中：增加该词条的计数值
                增加所有词条的计数值
            对每个类别：
                对每个词条：
                    将该词条的数目处于总词条数据得到条件概率
            返回每个类别的条件概率
"""
import numpy as np


class Bayes(object):
    """ 朴素贝叶斯算法类 """
    def __init__(self):
        pass

    def load_data_set(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

        class_vec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常文字
        return postingList, class_vec

    def create_vocab_list(self, data_set):
        """ 词汇列表:统计所有单词 """
        vocab_set = set([])
        for document in data_set:
            vocab_set = vocab_set | set(document)  # 并集
        return list(vocab_set)

    def word_2_vec(self, vocab_list, input_set):
        """ 构建词向量：每行是一条样本，每个单词是一个特征，0表示未出现，1表示出现 """
        return_vec = []
        for word_list in input_set:
            tmp = [0] * len(vocab_list)
            for word in word_list:
                if word in vocab_list:
                    tmp[vocab_list.index(word)] += 1
                else:
                    print("the word:%s is not in my vocabulary!" % word)
            return_vec.append(tmp)

        return return_vec

    def train_baye(self, train_matrix, train_label):
        """ 训练算法 """
        num_train_docs = len(train_matrix)  # 文档矩阵中文档的数据
        num_words = len(train_matrix[0])    # 词条向量的长度
        p_abusive = sum(train_label) / float(num_train_docs)  # 所有文档中属于类1所占的比例p(c=1), 先验概率
        p0_num = p1_num = np.ones(num_words)  # 创建一个长度为词条向量等长的列表
        p0_denom = p1_denom = 2.0
        # 遍历每一篇文档的词条向量
        for i in range(num_train_docs):
            if train_label[i] == 1:
                p1_num += train_matrix[i]
                p1_denom += sum(train_matrix[i])
            else:
                p0_num += train_matrix[i]
                p0_denom += sum(train_matrix[i])

        p1_vec = np.log(p1_num / p1_denom)
        p0_vec = np.log(p0_num / p0_denom)
        return p0_vec, p1_vec, p_abusive

    def classify_baye(self, vec_2_classify, p0_vec, p1_vec, p1_class):
        p1 = sum(vec_2_classify * p1_vec) + np.log(p1_class)
        p0 = sum(vec_2_classify * p0_vec) + np.log(1.0 - p1_class)
        if p1 > p0:
            return 1
        else:
            return 0


def test_baye():
    baye = Bayes()
    data, class_vec = baye.load_data_set()
    vocab = baye.create_vocab_list(data)
    train_mat = baye.word_2_vec(vocab, data)

    p0_v, p1_v, p_ab = baye.train_baye(train_mat, class_vec)
    test_entry = ['love', 'my']
    this_doc = np.array(baye.word_2_vec(vocab, test_entry))
    print('test_entry:%s classified as:', baye.classify_baye())


if __name__ == "__main__":
    test_baye()