#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
关联规则(一种推荐算法)：联分析是一种在大规模数据集中寻找有趣关系的一个工具集，有趣的关系可以分为：频繁项集和关联规则；可以分析通过购买某种物品会买其他哪种物品来进行推荐

名词解释：
    事务：一条交易认为是一个事务
    项：交易中的物品
    项集：多个项的集合
    频繁项集：在数据集合中频繁出现的项集，那么如何来定义或者量化是频繁的呢？可以用支持度来量化，具体是：项集在数据集中出现次数占数据集的总数目中的比例
    关联规则：是针对频繁项集而说的，比如有个频繁项集{P,H}，定义一条关联规则：p-》H的关联规则是有子集P可以很大程度上得到子集H，重点表现的是两者之间某种关系。
        同样该如何量化这个程度呢？可以用可信度或者置信度来量化啊，具体是：support(P|H)/support(P)，分子support(P|H)表示项集{P,H}的支持度。

Apriori原理：如果某个项集是频繁的，那么它的子项集也是频繁的；反之，如果某个项集非频繁，那么它的超集(包含它的项集)也是非频繁的
Apriori算法：
    1、统计满足最小支持度的大小为1的项
    2、两两合并组成大小为2的项集，计算支持度，选择满足最小支持度的项集
    3、重复进行
"""

import numpy as np


class AprioriAlgorithm:
    """Apriori算法实现"""
    def __init__(self):
        self.data = [[1, 3, 4],
                     [2, 3, 5],
                     [1, 2, 3, 5],
                     [2, 5]]

    def create_c1(self):
        # 构建大小为1的所有项
        c1 = []
        for transaction in self.data:
            for item in transaction:
                if [item] not in c1:
                    c1.append([item])
        c1.sort()
        return list(map(frozenset, c1))

    def scan_data(self, ck, min_support):
        """
        扫描数据集合，计算各个子项的次数，并去除小于最小支持度的项
        :param data_set: 数据集
        :param ck: 当前的子项
        :param min_support: 最小支持度
        :return:
        """
        ss_cnt = {}
        D = list(map(set, self.data))
        # 统计每个子项的次数
        for tid in D:
            for can in ck:
                if can.issubset(tid):  # can是tid的子集
                    if can not in ss_cnt:
                        ss_cnt[can] = 1
                    else:
                        ss_cnt[can] += 1

        # 计算支持度
        num_items = len(D)  # 总的项集数
        ret_list = []      # 满足最小支持度的项
        support_data = {}  # 所有子项和支持度
        for key in ss_cnt:
            support = ss_cnt[key] / num_items  # 计算每个子项的支持度
            if support >= min_support:
                ret_list.append(key)
            support_data[key] = support
        return ret_list, support_data

    def apriori_gen(self, lk, k):
        """
        创建ck
        :param lk: [[1,2,3]]
        :param k: 需要创建的ck
        :return:
        """
        ret_list = []
        len_lk = len(lk)
        for i in range(len_lk):
            for j in range(i+1, len_lk):  # 两两组合遍历
                # print('--------')
                # print('list(lk[i])=', list(lk[i]))
                # print('list(lk[j])=', list(lk[j]))
                # print('list(lk[i])=', list(lk[i]))
                # print('list(lk[j][:k-2])=', list(lk[j])[:k-2])
                l1 = list(lk[i])[:k-2]  # 关于k-2的疑惑书上解释得很清楚，为了避免重复操作
                l2 = list(lk[j])[:k-2]
                l1.sort()
                l2.sort()
                if l1 == l2:
                    ret_list.append(lk[i] | lk[j])
        return ret_list

    def apriori(self, min_support=0.5):
        c1 = self.create_c1()
        L1, support_data = self.scan_data(c1, min_support)
        L = [L1]  # 列表L会逐渐包含频繁项集L1,L2,L3...
        k = 2
        while len(L[k-2]) > 0:  # 当集合项中的个数大于0
            ck = self.apriori_gen(L[k-2], k)
            Lk, sup_k = self.scan_data(ck, min_support)
            support_data.update(sup_k)
            L.append(Lk)
            k += 1
        return L, support_data


class TreeNode:
    def __init__(self, name_value, name_occur, parent_node):
        self.name = name_value
        self.count = name_occur
        self.parent = parent_node
        self.node_link = None
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def disp(self, ind=1):
        """ 显示树 """
        print(' '*ind, self.name, ':', self.count)
        for child in self.children.values():
            child.disp(ind+1)


class FP_growth:
    """
    FP-group树来查找频繁项集，比Apriori要快很多(Apriori对于每个潜在的频繁项集都要扫描整个数据集，而FP-group树只扫描两次)
    基本过程：
        1、构建FP树：两次扫描：
            a、对所有元素项的出现次数进行统计
            b、第二次扫描只考虑频繁元素
        2、从FP树挖掘频繁项集
    """
    def load_data(self):
        data = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return data

    def create_init_set(self, data_set):
        ret_dict = {}
        for trans in data_set:
            ret_dict[frozenset(trans)] = 1
        return ret_dict

    def create_tree(self, data_set, min_sup=1):
        """
        构建FP树
        :param min_sup: 最小支持度，默认为1
        :return:
        """
        header_table = {}
        # 计算每个项的次数
        for trans in data_set:
            for item in trans:
                header_table[item] = header_table.get(item, 0) + data_set[trans]
        # 删除不满足最小支持度的元素项
        for k in header_table.keys():
            if header_table[k] < min_sup:
                del(header_table[k])
        # 没有元素退出
        freq_item_set = set(header_table.keys())
        if len(freq_item_set) == 0:
            return None, None

        for k in header_table:
            header_table[k] = [header_table[k], None]

        ret_tree = TreeNode('Null Set', 1, None)
        for trans, count in data_set.items():
            localD = {}
            for item in trans:
                if item in freq_item_set:
                    localD[item] = header_table[item][0]
            if len(localD) > 0:
                ordered_items = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
                self.update_tree(ordered_items, ret_tree, header_table, count)

        return ret_tree, header_table



if __name__ == "__main__":
    cls = AprioriAlgorithm()

    # L, support_data = cls.apriori()
    # print(L)
    # print(support_data)
    node = TreeNode('pyramid', 9, None)
    node.children['eye'] = TreeNode('eye', 13, None)
    node.disp()
