#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Process_data.py 
@Author  ：zly
@Date    ：2025/1/8 9:58 
'''
from PPF_all import *
train_x, train_y, test_x, test_y = [], [], [], []
model = SVC()
pred = PPF(train_x, train_y, test_x, test_y, model = model)