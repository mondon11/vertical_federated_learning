#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:56
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : vertical_lr_host.py

from vertical_logistic_regression.vertical_lr_base import ClientBase
import numpy as np

class ClientHost(ClientBase):
    """
    Host节点，只有X，没有Y
    """
    def __init__(self, X, config):
        super().__init__(config)
        #host训练数据X，没有Y
        self.X = X
        #host模型参数
        self.weights = np.zeros(X.shape[1])

    def send_data_to_guest(self, guest_name):
        """
        计算[[0.25*X_host*Θ_host]]和[[(X_host*Θ_host).^2]],发送给guest
        :param guest_name:guest节点名
        :return:none
        """
        public_key = self.data['public_key']
        z_host = np.dot(self.X, self.weights)
        u_host = 0.25 * z_host
        z_host_square = z_host ** 2
        encrypted_u_host = np.asarray([public_key.encrypt(x) for x in u_host])
        encrypted_z_host_square = np.asarray([public_key.encrypt(x) for x in z_host_square])
        #host节点数据新增[[0.25*X_host*Θ_host]]，为后续计算host的梯度
        self.data.update({'encrypted_u_host': encrypted_u_host})
        #发送[[0.25*X_host*Θ_host]]到guest，为后续guest计算梯度;
        #发送[[(X_host*Θ_host).^2]]到guest，为后续guest计算整个过程的loss
        data = {'encrypted_u_host': encrypted_u_host, 'encrypted_z_host_square':encrypted_z_host_square}
        self.send_data(data, self.other_client[guest_name])

    def send_data_to_arbiter(self, arbiter_name):
        """
        计算host侧的加密后的梯度，添加random mask后发送给arbiter
        :param arbiter_name: arbiter节点名
        :return: none
        """
        encrypted_u_host = self.data['encrypted_u_host']
        encrypted_u_guest = self.data['encrypted_u_guest']
        encrypted_dJ_host = (np.dot(self.X.T,(encrypted_u_host + encrypted_u_guest)) +
                             self.config['lambda'] * self.weights) / self.X.shape[0]
        random_mask = np.random.rand(self.X.shape[1])
        encrypted_masked_dJ_host = encrypted_dJ_host + random_mask
        self.data.update({'random_mask':random_mask})
        data = {'encrypted_masked_dJ_host':encrypted_masked_dJ_host}
        self.send_data(data, self.other_client[arbiter_name])

    def update_weights(self):
        """
        host侧更新模型系数
        :return: none
        """
        dJ_host = self.data['masked_dJ_host'] - self.data['random_mask']
        self.weights = self.weights - self.config['lr'] * dJ_host
        print('host weights: ' + self.weights.__str__())



