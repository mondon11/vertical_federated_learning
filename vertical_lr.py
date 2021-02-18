#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:59
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : vertical_lr.py

from util import *
from vertical_logistic_regression.vertical_lr_arbiter import ClientArbiter
from vertical_logistic_regression.vertical_lr_host import ClientHost
from vertical_logistic_regression.vertical_lr_guest import ClientGuest
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
import time
import pickle

def vertical_lr_train(X_host_train, X_guest_train, Y_train, config):
    """
    逻辑回归纵向联邦学习训练
    :param X_host_train: host侧训练集特征
    :param X_guest_train: guest侧训练集特征
    :param Y_train: 训练集的标签集
    :param config: 训练参数
    :return: none
    """
    print('X_host_train:', X_host_train.shape, '   X_guest_train:', X_guest_train.shape)

    client_host = ClientHost(X_host_train, config)
    client_guest = ClientGuest(X_guest_train, Y_train, config)
    client_arbiter = ClientArbiter(config)

    client_host.connect('client_guest', client_guest)
    client_host.connect('client_arbiter', client_arbiter)
    client_guest.connect('client_host', client_host)
    client_guest.connect('client_arbiter', client_arbiter)
    client_arbiter.connect('client_host', client_host)
    client_arbiter.connect('client_guest', client_guest)

    with open('./result/vlr_train_loss.txt', 'w') as f:
        f.write(
            'epoch' + '\t' + 'time' + '\t' +  'vlr_train_loss' + '\n')
        f.flush()
        for i in range(config['n_iter']):
            client_arbiter.generate_send_key('client_host', 'client_guest')
            client_host.send_data_to_guest('client_guest')
            client_guest.send_data_to_host('client_host')
            client_host.send_data_to_arbiter('client_arbiter')
            client_guest.send_data_to_arbiter('client_arbiter')
            client_arbiter.send_dJ_to_host_and_guest('client_host', 'client_guest')
            client_host.update_weights()
            client_guest.update_weights()

            f.write(str(i) + '\t' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\t' + str(client_arbiter.loss[-1]) + '\n')
            f.flush()

    with open('./model/host_weights.pkl' , 'wb') as f:
        pickle.dump(client_host.weights, f)

    with open('./model/guest_weights.pkl', 'wb') as f:
        pickle.dump(client_guest.weights, f)

def vertical_lr_test(X_host_test, X_guest_test, Y_test):
    """
    逻辑回归纵向联邦学习测试
    :param X_host_test: host侧测试集特征
    :param X_guest_test: guest侧测试集特征
    :param Y_test: 测试集的标签集
    :return: none
    """
    with open('./model/host_weights.pkl' , 'rb') as f:
        host_weights = pickle.load(f)
    with open('./model/guest_weights.pkl' , 'rb') as f:
        guest_weights = pickle.load(f)

    host_test_predict_prob = np.dot(X_host_test, host_weights)
    guest_test_predict_prob = np.dot(X_guest_test, guest_weights)
    vlr_predict_score = 1 / (1 + np.exp(-(host_test_predict_prob + guest_test_predict_prob)))
    vlr_predict_auc = roc_auc_score(Y_test, vlr_predict_score)
    vlr_predict_acc = accuracy_score(Y_test, (vlr_predict_score >= 0.5).astype(int))

    print('*****vlr_predict_auc: ',vlr_predict_auc,'*****')
    print('*****vlr_predict_acc: ', vlr_predict_acc, '*****')

    with open('./result/vlr_test_result.txt', 'w') as f:
        f.write('time' + '\t' +'vlr_predict_auc' + '\t' + 'vlr_predict_acc' + '\n')
        f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\t' + str(vlr_predict_auc) + '\t' + str(vlr_predict_acc))




if __name__ == '__main__':
    config = {
    'n_iter': 100,
    'lambda': 10,
    'lr': 0.05,
    'host_index': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    'guest_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    X_train, X_test, Y_train, Y_test = load_data()
    X_host_train, X_guest_train, X_host_test, X_guest_test = vertically_split_data(X_train, X_test, config['host_index'], config['guest_index'])

    #训练
    vertical_lr_train(X_host_train, X_guest_train, Y_train, config)

    #测试
    vertical_lr_test(X_host_test, X_guest_test, Y_test)

