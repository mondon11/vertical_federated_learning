#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:59
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : local_lr.py

from util import *
import numpy as np
import time
import pickle
from sklearn.metrics import roc_auc_score,accuracy_score

def local_lr_train(X_train, Y_train, config):
    """
    逻辑回归训练
    :param X_train: 训练集特征
    :param Y_train: 训练集标签
    :param config: 训练参数
    :return: none
    """
    print('X_train:', X_train.shape, '   Y_train:', Y_train.shape)
    weights = np.zeros(X_train.shape[1])
    loss_list = []
    with open('./result/local_lr_train_loss.txt', 'w') as f:
        f.write('epoch' + '\t' + 'time' + '\t' +  'local_lr_train_loss' + '\n')
        f.flush()
        for i in range(config['n_iter']):
            z = np.dot(X_train, weights)
            h_z = 1 / (1 + np.exp(- z))
            loss = (-1 / X_train.shape[0]) * (Y_train.T.dot(np.log(h_z)) + (1 - Y_train).T.dot(np.log(1 - h_z))) + config['lr'] * weights.T.dot(weights) / (2 * X_train.shape[0])
            loss_list.append(loss)
            grad = (1 / X_train.shape[0]) * (X_train.T.dot(h_z - Y_train)) + (config['lambda'] / X_train.shape[0]) * weights
            weights = weights - config['lr'] * grad
            print('******local_LR loss: ', loss, '******')
            f.write(str(i) + '\t' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\t' + str(loss_list[-1]) + '\n')
            f.flush()

    with open('./model/local_lr_weights.pkl' , 'wb') as f:
        pickle.dump(weights, f)

def local_lr_test(X_test, Y_test):
    """
    逻辑回归测试
    :param X_test: 测试集特征
    :param Y_test: 测试集标签
    :return: none
    """
    with open('./model/local_lr_weights.pkl' , 'rb') as f:
        weights = pickle.load(f)
    predict_prob = np.dot(X_test, weights)
    predict_score = 1 / (1 + np.exp(-1 * predict_prob))
    predict_auc = roc_auc_score(Y_test, predict_score)
    predict_acc = accuracy_score(Y_test, (predict_score >= 0.5).astype(int))

    print('*****local_lr_predict_auc: ', predict_auc, '*****')
    print('*****local_lr_predict_acc: ', predict_acc, '*****')

    with open('./result/local_lr_test_result.txt', 'w') as f:
        f.write('time' + '\t' +'local_lr_predict_auc' + '\t' + 'local_lrpredict_acc' + '\n')
        f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\t' + str(predict_auc) + '\t' + str(predict_acc))

if __name__ == '__main__':
    config = {
    'n_iter': 100,
    'lambda': 10,
    'lr': 0.05
    }
    X_train, X_test, Y_train, Y_test = load_data()

    #训练
    local_lr_train(X_train, Y_train, config)

    #测试
    local_lr_test(X_test, Y_test)


