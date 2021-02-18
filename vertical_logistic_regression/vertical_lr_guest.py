#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:56
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : vertical_lr_guest.py

from vertical_logistic_regression.vertical_lr_base import ClientBase
import numpy as np
import math

class ClientGuest(ClientBase):
    """
    guest节点，有X和Y
    """
    def __init__(self, X, Y, config):
        super().__init__(config)
        #guest训练数据X和Y
        self.X = X
        self.Y = Y
        #guest模型参数
        self.weights = np.zeros(X.shape[1])

    def send_data_to_host(self, host_name):
        """
        计算[[0.25*X_guest*Θ_guest-Y-0.5]],发送给host
        :param host_name: host节点名
        :return: none
        """
        public_key = self.data['public_key']
        z_guest = np.dot(self.X, self.weights)
        u_guest = 0.25 * z_guest - self.Y +0.5
        encrypted_u_guest = np.asarray([public_key.encrypt(x) for x in u_guest])
        #guest节点数据新增X_guest*Θ_guest和[[0.25*X_guest*Θ_guest-Y-0.5]],为后续guest计算梯度
        self.data.update({'z_guest':z_guest, 'encrypted_u_guest':encrypted_u_guest})
        #发送[[0.25*X_guest*Θ_guest-Y-0.5]]到host，为host后续计算梯度
        data = {'encrypted_u_guest':encrypted_u_guest}
        self.send_data(data, self.other_client[host_name])

    def send_data_to_arbiter(self, arbiter_name):
        """
        计算guest侧的加密后的梯度，添加random mask后发送给arbiter;计算host和guest侧的总loss,发送给arbiter解密
        :param arbiter_name: arbiter节点名
        :return: none
        """
        encrypted_u_host = self.data['encrypted_u_host']
        encrypted_u_guest = self.data['encrypted_u_guest']
        encrypted_dJ_guest = (np.dot(self.X.T, (encrypted_u_host + encrypted_u_guest)) +
                              self.config['lambda'] * self.weights) / self.X.shape[0]
        random_mask = np.random.rand(self.X.shape[1])
        encrypted_masked_dJ_guest = encrypted_dJ_guest + random_mask
        self.data.update({'random_mask': random_mask})
        e_vec = np.ones(self.X.shape[0])
        encrypted_loss = (np.dot((0.5 * e_vec - self.Y).T, (4 * encrypted_u_host + self.data['z_guest']))  + \
                         self.X.shape[0] * math.log(2) + np.dot(0.125 * e_vec.T, self.data['encrypted_z_host_square']) + \
                         np.dot(0.125 * e_vec, (self.data['z_guest'] * (8 * self.data['encrypted_u_host'] + self.data['z_guest'])))) / self.X.shape[0]
        data = {'encrypted_masked_dJ_guest': encrypted_masked_dJ_guest, 'encrypted_loss':encrypted_loss}
        self.send_data(data, self.other_client[arbiter_name])

    def update_weights(self):
        """
        guest侧更新模型系数
        :return: none
        """
        dJ_guest = self.data['masked_dJ_guest'] - self.data['random_mask']
        self.weights = self.weights - self.config['lr'] * dJ_guest
        print('guest weights: ' + self.weights.__str__())