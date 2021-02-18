#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:59
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : util.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    加载数据
    :return: X_train, X_test, Y_train, Y_test
    """
    breast = load_breast_cancer()
    X_train, X_test, Y_train, Y_test = train_test_split(breast.data, breast.target, random_state=1)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.fit_transform(X_test)
    return X_train, X_test, Y_train, Y_test

def vertically_split_data(X_train, X_test, host_index, guest_index):
    """
    纵向切分数据（特征）
    :param X_train: 训练集特征
    :param X_test: 测试集特征
    :param host_index: host侧特征索引
    :param guest_index: guest侧特征索引
    :return: X_host_train, X_guest_train, X_host_test, X_guest_test
    """
    X_host_train = X_train[:, host_index]
    X_guest_train = X_train[:, guest_index]
    X_host_test = X_test[:, host_index]
    X_guest_test = X_test[:, guest_index]
    return X_host_train, X_guest_train, X_host_test, X_guest_test
