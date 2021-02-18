#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:56
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : vertical_lr_arbiter.py

from vertical_logistic_regression.vertical_lr_base import ClientBase
from phe import paillier
import numpy as np

class ClientArbiter(ClientBase):
    """
    可信第三方
    """
    def __init__(self, config):
        super().__init__(config)
        #公钥
        self.public_key = None
        #私钥
        self.private_key = None
        #计算过程中的损失函数值
        self.loss = []

    def generate_send_key(self, host_name, guest_name):
        """
        arbiter生成公钥和私钥，发给host和guest
        :param host_name: host节点名
        :param guest_name: guest节点名
        :return: none
        """
        public_key, private_key = paillier.generate_paillier_keypair()
        self.public_key = public_key
        self.private_key = private_key
        data = {'public_key': public_key}
        self.send_data(data, self.other_client[host_name])
        self.send_data(data, self.other_client[guest_name])

    def send_dJ_to_host_and_guest(self, host_name, guest_name):
        """
        解密host和guest侧的梯度，发回给他们，为后续更新梯度;解密总loss
        :param host_name: host节点名
        :param guest_name: guest节点名
        :return: none
        """
        encrypted_masked_dJ_host = self.data['encrypted_masked_dJ_host']
        encrypted_masked_dJ_guest = self.data['encrypted_masked_dJ_guest']
        encrypted_loss = self.data['encrypted_loss']
        masked_dJ_host = np.asarray([self.private_key.decrypt(x) for x in encrypted_masked_dJ_host])
        masked_dJ_guest = np.asarray([self.private_key.decrypt(x) for x in encrypted_masked_dJ_guest])
        loss = self.private_key.decrypt(encrypted_loss)
        self.loss.append(loss)
        print('*****VFL-LR: ',loss,'*****')
        data_to_host = {'masked_dJ_host':masked_dJ_host}
        data_to_guest = {'masked_dJ_guest':masked_dJ_guest}
        self.send_data(data_to_host, self.other_client[host_name])
        self.send_data(data_to_guest, self.other_client[guest_name])


