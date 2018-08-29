# -*- coding:utf-8 -*-
from math import log
from random import sample


class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # 对于real value的条件为<，对于类别值得条件为=
        # 将满足条件的放入左树
        self.real_value_feature = True
        self.conditionValue = None
        self.leafNode = None
        pass

    def get_predict_value(self, instance):
        # 到达叶子节点
        if self.leafNode:
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError("the tree is null")
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info


class LeafNode:
    '''
    叶子节点
    '''
    def __init__(self, idset):
        # 叶子节点的样本集合
        self.idset = idset
        # 该叶子节点的预测值
        self.predictValue = None
        pass

    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, targets, loss):
        self.predictValue = loss.update_ternimal_regions(targets, self.idset)
        pass
    pass


def MSE(values):
    """
    均平方误差 mean square error
    """
    if len(values) < 2:
        return 0
    mean = sum(values) / float(len(values))
    error = 0.0
    for v in values:
        error += (mean-v)*(mean-v)
    return error


def FriedmanMSE(left_values, right_values):
    """
    参考Friedman的论文Greedy Function Approximation: A Gradient Boosting Machine中公式35
    """
    # 假定每个样本的权重都为1
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right))


def construct_decision_tree(dataset, remainedSet, targets, depth, leaf_nodes, max_depth, loss, criterion='MSE', split_points=0):
    '''
    构建集成学习器的基学习器
    :param dataset: 全集
    :param remainedSet: 剩下的集合
    :param targets: 目标残差，也是标签但会更新
    :param depth: 目前树的深度，用于剪枝
    :param leaf_nodes: 叶子结点
    :param max_depth: 最大深度
    :param loss: 损失函数
    :param criterion: 
    :param split_points: 分割的特征
    :return: 
    '''
    if depth < max_depth:
        # todo 通过修改这里可以实现选择多少特征训练
        attributes = dataset.get_attributes()
        mse = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        # 遍历所有特征，寻找方差最大的
        for attribute in attributes:
            is_real_type = dataset.is_real_type_field(attribute)
            attrValues = dataset.get_distinct_valueset(attribute)
            # 只有当时实数值，并且需要划分多个点时，通过采样值，使得变为二分类问题
            if is_real_type and split_points > 0 and len(attrValues) > split_points:
                attrValues = sample(attrValues, split_points)
            for attrValue in attrValues:
                leftIdSet = []
                rightIdSet = []
                # 对剩下的集合寻找最优划分点
                for Id in remainedSet:
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    # 将满足条件的放入左子树
                    if (is_real_type and value < attrValue) or (not is_real_type and value == attrValue):
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                sum_mse = MSE(leftTargets) + MSE(rightTargets)
                # 记录MSE最小的最为划分点
                if mse < 0 or sum_mse < mse:
                    selectedAttribute = attribute
                    conditionValue = attrValue
                    mse = sum_mse
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
        if not selectedAttribute or mse < 0:
            raise ValueError("cannot determine the split attribute.")
        # 创建树
        tree = Tree()
        tree.split_feature = selectedAttribute
        tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
        tree.conditionValue = conditionValue
        # 递归，对于每个分类器标签不变，对于不同分类器，target标签改变
        tree.leftTree = construct_decision_tree(dataset, selectedLeftIdSet, targets, depth+1, leaf_nodes, max_depth, loss)
        tree.rightTree = construct_decision_tree(dataset, selectedRightIdSet, targets, depth+1, leaf_nodes, max_depth, loss)
        return tree
    else:  # 是叶子节点
        node = LeafNode(remainedSet)
        node.update_predict_value(targets, loss)
        leaf_nodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree
