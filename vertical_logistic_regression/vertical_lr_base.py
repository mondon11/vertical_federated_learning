#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午1:55
# @Author  : jt_hou
# @Email   : 949241101@qq.com
# @File    : vertical_lr_base.py

class ClientBase:
    def __init__(self, config):
        #模型参数
        self.config = config
        #中间结果
        self.data = {}
        #其他节点信息
        self.other_client = {}

    def connect(self, client_name, target_client):
        """
        与其他节点建立连接
        :param client_name: 本节点名
        :param target_client: 目标节点
        :return: none
        """
        self.other_client[client_name] = target_client

    def send_data(self, data, target_client):
        """
        向目标节点发送数据
        :param data: 发送的数据
        :param target_client: 目标节点
        :return: none
        """
        target_client.data.update(data)