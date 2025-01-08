from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from matplotlib import pyplot as plt
from numpy import *
import numpy as np
import pandas as pd
from collections import Counter
import copy
import math
import random
import sys
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from imblearn.metrics import geometric_mean_score as G_mean
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from os import listdir
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import gaussian_process, kernel_approximation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class svm_problem:
    def __init__(self, x, y):
        self.l = np.shape(y)[0]
        self.x = x  # 样本
        self.r_y = copy.deepcopy(y)  # 真实标签
        self.y = copy.deepcopy(y)
        self.pos = [1] * self.l  # positive flag  pos:1 neg:0
        self.use = [1] * self.l


class predict_result:
    def __init__(self, x, y):
        self.l = np.shape(y)[0]
        self.x = x
        self.r_y = copy.deepcopy(y)  # 真实标签
        self.y = copy.deepcopy(y)
        self.p = [NAN] * len(y)
        self.r_y_p = [NAN] * len(y)


class glo_param:
    # 类分解过程中的一些变量
    useClusterNum = 0  # 保留的簇的个数
    avgSubclass = 0  # 子簇的平均大小
    tK = 3  # 采样的近邻大小
    sub_count = []
    sub_cluster = []
    sub_c = []
    hashMap = {}  # 记录父标签对应的子标签
    oriLabelIdx = []  # 记录初始model中正负类的位置

    # 指标相关
    tp = [0] * 500  # 等于maxnum
    fp = [0] * 500
    tn = [0] * 500
    fn = [0] * 500
    svm_train_f1 = 0
    svm_train_gm = 0
    kmean_train_f1 = 0
    kmean_train_gm = 0

    def __init__(self, prob, k_prob, model):
        self.prob = prob
        self.k_prob = k_prob
        self.model = model


class sortSample:
    def __init__(self):
        self.id = -1
        self.d = -1


# 定义一个隐藏print输出的类，用于屏蔽聚类的提示信息的输出
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class Cssc:
    def __init__(self, maxnum=500, weight=0.7):
        self.maxnum = maxnum  # 簇的最大分裂个数
        self.weight = weight

    def fit(self, x, y):
        self.x = x  # x
        self.y = y  # y
        self.label = list(set(y))  # 所有的标签
        self.count = [0] * len(set(y))  # 各个簇中训练样本的数量
        self.old_count = [0] * len(set(y))
        self.count_acc = [0] * len(set(y))  # 各个簇中预测正确的训练样本的数量
        self.old_count_acc = [0] * len(set(y))
        self.cluster_acc = [NAN] * len(set(y))  # 簇预测的正确率
        self.old_cluster_acc = [NAN] * len(set(y))
        self.sse = [0] * len(set(y))  # 簇的密集程度
        self.old_sse = [0] * len(set(y))  # 簇的密集程度
        self.cluster = [NAN] * len(y)  # 训练样本属于的簇序号
        self.old_yc = [NAN] * len(set(y))
        self.kmeans_noise_flag = [False] * len(y)  # 样本是否是噪声
        self.c = [NAN] * len(y)  # 训练样本预测标签

        self.d = [0] * self.maxnum
        # 下面需要注意下 ，别弄错
        self.use = [1] * len(y)  # 这个样本是否使用
        self.use_k = [1] * self.maxnum  # 这个簇是否使用

        # 别弄错
        # 记录该样本是否是正类
        pos = copy.deepcopy(y)
        pos[pos == -1] = 0
        self.pos = pos  #
        pos_model = np.array(copy.deepcopy(list(set(y))))
        pos_model[pos_model == -1] = 0
        self.pos_model = pos_model  # 类分解后标签对应的是否是正类

        self.size = [0] * len(set(y))  # 簇的样本数量 训练集+测试集
        self.u = [NAN] * len(y)  # 训练样本的预测概率
        self.sse_labeled = [0] * len(set(y))  # 簇的密集程度 有标签
        self.sse_labeled_acc = [0] * len(set(y))
        self.train_acc = 0  # 训练样本的准确率
        self.avg_cluster_acc = 0  # 每个簇的平均准确率

        self.total_sse = 0

        self.parent = [-1] * self.maxnum
        # // 是否新分裂的簇
        # 0 - 不是；1 - 是；2 - 被新簇覆盖的旧的簇
        self.new_split = [0] * self.maxnum
        self.need_split = [False] * self.maxnum

        self.old_cluster = NAN  # 上一次训练样本所属的簇序号
        self.old_c = NAN  # 上一次训练样本的预测标签
        self.old_c1 = NAN  # 上一次测试样本的预测标签
        self.old_k = NAN

        self.history_acc = [0] * 100
        self.history_SSE = [0] * 100
        self.history_FVP = [0] * 100

        # 一些簇的相关指标
        self.k = len(set(y))  # 质心的个数
        self.k_c = [0] * len(set(y))  # 每个簇的样本个数
        self.y_c = [NAN] * len(set(y))  # 簇对应的类别
        self.x_c = [NAN] * len(set(y))  # 每个簇的质心
        self.Center = NAN  # 旧的质心
        self.w = [0] * self.maxnum  # 计算距离时候的权重
        self.subclass_y = [NAN] * self.maxnum  # 作为子类的标签

        self.size = [NAN] * self.maxnum  # 该簇的样本数量：训练集+测试集
        # positive flag  pos:1 neg:0

        # 根据第二次更改新增加的变量
        self.train_f1 = NAN
        # 根据需要将svm的model放入kmeans中
        # self.model_svm = NAN
        # 计算每个质心的平均距离
        self.Dis = NAN
        self.history_DIS = [0] * 100
        self.cluster_gmeans = [NAN] * len(set(y))  # 簇预测的gmeans
        self.old_cluster_gmeans = [NAN] * len(set(y))

    # 初始化质心
    def k_center(self):
        x, y = self.x, self.y
        # print("``")
        # print(len(y))
        # print(range(len(K_data.label)),range(len(y)))
        for i in range(self.k):
            self.k_c[i] = 0
            self.y_c[i] = self.label[i]
        for i in range(self.k):  # 对于每一个簇
            if self.use[i] == 0: continue
            sum = 0
            sum_num = 0
            for j in range(len(y)):  # 对于每一个样本
                if y[j] == self.label[i]:  # 如果样本的标签和簇的标签一样的话

                    self.k_c[i] += 1  # 簇中的样本个数+1
                    self.count[i] += 1  # 簇中的训练样本个数+1
                    self.cluster[j] = i  # 训练样本所属的簇序号
                    self.c[j] = y[j]
                    sum += x[j]
                    sum_num += 1
            self.x_c[i] = sum / sum_num
            # print(K_data.x_c[i])
        print("train class num :")
        for i in range(self.k):  # 对于每一个簇
            print("label({}):{}\t".format(self.label[i], self.k_c[i]), end="\t")
            x_ls = x[y == self.label[i]]
            # print(x_ls.shape)
            sse_this = 0
            for x2 in x_ls:
                # print(np.sqrt(np.sum(np.square(x2 - K_data.x_c[i]))))
                sse_this += distance(x2, self.x_c[i]) ** 2
            self.sse[i] = sse_this / x_ls.shape[0]
            print("sse:", self.sse[i])
            self.count_acc[i] = self.count[i]
            self.cluster_acc[i] = 1

    def kmean_train(self, test_k):
        # 记录中间结果文件
        # f_U = open("detail_cluster.txt", "a+", encoding='utf-8')
        # f_U.writelines("{}\n".format(train_k.Data_name))
        update_del = 100
        need_split = [False] * self.maxnum
        sum_SSE = sum_ACC = 0
        sum_FVP = 0
        sum_DIS = 0
        split_flg = True
        restore = False
        weight = self.weight
        history_acc = self.history_acc
        history_SSE = self.history_SSE
        history_FVP = self.history_FVP
        history_DIS = self.history_DIS
        n = 0
        while n == 0 or (split_flg == True and self.k < self.maxnum):
            print("当前的n为：%d\n" % n)
            #         保存上次的聚类结果
            self.old_cluster = copy.deepcopy(self.cluster)  # 训练样本属于的簇序号
            self.old_c = copy.deepcopy(self.c)  # 训练样本的预测标签

            # test上的上次的预测结果
            test_k.old_c1 = copy.deepcopy(test_k.c1)
            self.kmean_predict0()
            self.kmean_predict_testdata0(test_k)
            history_acc[n] = self.train_acc
            history_SSE[n] = self.total_sse
            history_FVP[n] = self.train_f1
            history_DIS[n] = self.Dis
            sum_ACC += history_acc[n]
            sum_SSE += history_SSE[n]
            sum_FVP += history_FVP[n]
            sum_DIS += history_DIS[n]
            # // // // // // // // // // / 保存上次的质心相关信息 // // // // // // // // // // // // // // // // // // // // // // // //
            self.old_k = copy.deepcopy(self.k)

            self.Center = copy.deepcopy(self.x_c)  # 保留分裂前的质心

            self.old_yc = copy.deepcopy(self.y_c)  # //保存分裂前的质心标签
            self.old_sse = copy.deepcopy(self.sse)
            self.old_cluster_acc = copy.deepcopy(self.cluster_acc)

            # // // // // // // // 第三步：分裂 // // // // // // // // // // // //
            if self.train_acc >= 1:
                print("train_acc 为1跳出")
                break
            dif = 0
            for i in range(test_k.x.shape[0]):  # // 统计两次预测的差异
                if test_k.old_c1[i] != test_k.c1[i]:
                    dif += 1
            if n != 0 and dif < 1:
                print("两个模型预测test保持一样")
                break

            split_flg = False
            # // 初始化所有簇的分裂状态
            self.parent = [-1] * self.maxnum
            # // 是否新分裂的簇
            # 0 - 不是；1 - 是；2 - 被新簇覆盖的旧的簇
            self.new_split = [0] * self.maxnum
            self.need_split = [False] * self.maxnum

            # // 找出需要分裂的簇
            i = 0
            while i < self.old_k:
                # print("while")
                if self.cluster_acc[i] < 0.9 and self.k < self.maxnum:
                    self.need_split[i] = True
                    print("\t\tcluster_acc[%d]:%f\tn:%d\n" % (i, self.cluster_acc[i], n))
                    if self.kmean_split1_1(test_k, i) == True:
                        split_flg = True
                i += 1
            del i
            if split_flg == True:
                self.kmean_predict0()
            # 考虑是否还原
            if n != 0:
                history_acc[n + 1] = self.train_acc
                history_SSE[n + 1] = self.total_sse
                history_FVP[n + 1] = self.train_f1
                history_DIS[n + 1] = self.Dis

                # d = (weight * history_FVP[n] / sum_FVP - (1 - weight) * history_SSE[n] / sum_SSE) - weight * (
                #         history_FVP[n + 1] / sum_FVP - (1 - weight) * history_SSE[n + 1] / sum_SSE)
                # d1 = - history_SSE[n] / sum_SSE + history_FVP[n] / sum_FVP + history_DIS[n] / sum_DIS
                # d2 = - history_SSE[n + 1] / sum_SSE + history_FVP[n + 1] / sum_FVP + history_DIS[n + 1] / sum_DIS
                # d1 = -(1 - train_k.weight) * history_SSE[n] / sum_SSE + train_k.weight * history_FVP[n] / sum_FVP
                # d2 = - (1 - train_k.weight) * history_SSE[n + 1] / sum_SSE + train_k.weight * history_FVP[n + 1] / sum_FVP

                d1 = (1 - self.weight) * history_acc[n] / sum_ACC + self.weight * history_FVP[n] / sum_FVP
                d2 = (1 - self.weight) * history_acc[n + 1] / sum_ACC + self.weight * history_FVP[n + 1] / sum_FVP
                # 记录中间过程
                d = history_FVP[n] - history_FVP[n + 1]
                # f_U.writelines("{}\t{}\t{}\t{}\n".format(history_acc[n] / sum_ACC,history_FVP[n] / sum_FVP,history_acc[n + 1] / sum_ACC,history_FVP[n + 1] / sum_FVP))
                if d > 0:
                    print("\n Restore former clusters!")
                    self.k = copy.deepcopy(self.old_k)
                    self.x_c = copy.deepcopy(self.Center)

                    self.y_c = copy.deepcopy(self.old_yc)
                    self.kmean_predict0()
                    self.kmean_predict_testdata0(test_k)
            n += 1
            print("本轮簇个数为:%d\n" % self.k)

        self.kmean_predict0()
        self.kmean_predict_testdata0(test_k)
        print("依次考察新分裂的簇")
        # todo
        i = 0
        # while i < self.k:  # // 依次考察新分裂的簇，是否需要撤销
        #     if self.new_split[i] == 1:  # //是否新分裂的簇 0-不是；1-是；2-被新簇覆盖的旧的簇
        #         # //如果新分裂的簇，他的准确率小于平均值（训练样本中各个簇的平均准确率）
        #         if self.cluster_acc[i] < self.avg_cluster_acc and self.y_c[i] == -1:
        #             # //要考虑两个条件（1.这个簇是不是新分裂出来的簇.2.这个簇的准确率是不是小于平均值）
        #             if self.kmean_delete_centroid(i):
        #                 i -= 1
        #                 restore = True
        #     i += 1
        del i
        # self.iterative_update2_1(test_k, 0)

    # 多分类
    def kmean_split1(train_k, test_k, no):
        #
        zly_old_old = 0
        zly_old_new = 0

        if train_k.count[no] <= 1:
            kmean_delete_centroid(train_k, no)
            train_k.old_k -= 1
            return True
        del_ = 0
        n = 0
        flg = False
        num = [0] * len(train_k.label)
        sum_all = []
        cluster_size = copy.deepcopy(train_k.count[no])
        for k in range(len(train_k.label)):  # //判断属于哪个类别
            sum_ = 0
            num[k] = 0
            for j in range(train_k.x.shape[0]):  # //计算该簇中各个类的样本的“质心”
                if train_k.cluster[j] == no and train_k.y[j] == train_k.label[k]:
                    num[k] += 1
                    sum_ += train_k.x[j]
            # print(sum_)
            sum_all.append(sum_)
        print("Split cluster %d: " % no)
        del_flg = False

        for k in range(len(train_k.label)):
            print("  Label-%d: %d" % (train_k.label[k], num[k]))
            if train_k.label[k] == train_k.y_c[no]:
                if num[k] > 0:

                    train_k.x_c[no] = sum_all[k] / num[k]
                    train_k.parent[no] = no
                    train_k.new_split[no] = 2
                    print("预测正确的样本分为一个簇(no:%d->%d,num[label:%d]:%d)\n" % (no, no, train_k.label[k], num[k]))
                    zly_old_old += 1
                else:
                    del_flg = True
            elif (num[k] > 2 or double(num[k]) / cluster_size > 0.05) and train_k.k < train_k.maxnum:
                flg = True
                # train_k.x_c[train_k.k]=(sum_all[k] / num[k])
                train_k.x_c.append(sum_all[k] / num[k])
                train_k.y_c.append(train_k.label[k])
                train_k.parent[train_k.k] = no
                train_k.new_split[train_k.k] = 1
                train_k.k += 1
                print("预测不正确的样本分为一个簇(no:%d->%d,num[label:%d]:%d)\n" % (
                    no, train_k.k, train_k.label[k], num[k]))
                zly_old_new += 1
        if del_flg:  # //如果正确的训练样本数量为0，则把第no个簇（本簇）覆盖掉
            train_k.x_c[no] = copy.deepcopy(train_k.x_c[train_k.k - 1])
            train_k.y_c[no] = copy.deepcopy(train_k.y_c[train_k.k - 1])
            train_k.parent[no] = 1
            train_k.new_split[no] = 1
            train_k.k -= 1
            print("\t%d簇没有正确的训练样本，取消该簇\t" % no)
        print("%d簇被分成了：%d（%d（属于该簇的样本为一个簇）+%d（其他样本簇的数量））\n" % (
            no, zly_old_old + zly_old_new, zly_old_old, zly_old_new));
        return True

    # 质心迭代
    def iterative_update2_1(self, test_k, n):
        train_k = self
        n1 = n + 1
        update_del = inf
        weight = train_k.weight
        history_acc = train_k.history_acc
        history_SSE = train_k.history_SSE
        history_FVP = train_k.history_FVP
        history_DIS = train_k.history_DIS
        tmp_sum_SSE, tmp_sum_ACC, tmp_sum_FVP = NAN, NAN, NAN
        tmp_sum_DIS = NAN
        kmean_predict0(train_k)

        kmean_predict_testdata0(train_k, test_k)
        tmp_sum_ACC = history_acc[n] = train_k.train_acc
        tmp_sum_SSE = history_SSE[n] = train_k.total_sse
        tmp_sum_FVP = history_FVP[n] = train_k.train_f1
        tmp_sum_DIS = history_DIS[n] = train_k.Dis
        print("开始质心迭代！")

        while update_del > 0.001:  # //当质心不再发生变化
            # //保存旧的质心
            tmp_Center = copy.deepcopy(train_k.x_c)
            update_del = kmean_update_1(train_k, test_k)
            print(update_del)
            kmean_predict0(train_k)
            kmean_predict_testdata0(train_k, test_k)
            history_acc[n1] = train_k.train_acc
            history_SSE[n1] = train_k.total_sse
            history_FVP[n1] = train_k.train_f1
            history_DIS[n1] = train_k.Dis
            tmp_sum_ACC += history_acc[n1]
            tmp_sum_SSE += history_SSE[n1]
            tmp_sum_FVP += history_FVP[n1]
            tmp_sum_DIS += history_DIS[n1]
            # t1 = -history_SSE[n1 - 1] / tmp_sum_SSE + history_FVP[n1 - 1] / tmp_sum_FVP + history_DIS[n1 - 1] / tmp_sum_DIS
            # t2 = -history_SSE[n1] / tmp_sum_SSE + history_FVP[n1] / tmp_sum_FVP + history_DIS[n1] / tmp_sum_DIS
            # t1 = -(1 - train_k.weight) * history_SSE[n1 - 1] / tmp_sum_SSE + train_k.weight * history_FVP[
            #     n1 - 1] / tmp_sum_FVP
            # t2 = -(1 - train_k.weight) * history_SSE[n1] / tmp_sum_SSE + train_k.weight * history_FVP[n1] / tmp_sum_FVP
            t1 = (1 - train_k.weight) * history_acc[n1 - 1] / tmp_sum_ACC + train_k.weight * history_FVP[
                n1 - 1] / tmp_sum_FVP
            t2 = (1 - train_k.weight) * history_acc[n1] / tmp_sum_ACC + train_k.weight * history_FVP[n1] / tmp_sum_FVP

            print("update_del:%f,t1:%f\t t2:%f" % (update_del, t1, t2))
            if (t1 >= t2):
                print("撤销本次质心迭代，恢复原质心!")
                train_k.x_c = copy.deepcopy(tmp_Center)
                break
            n1 += 1

    # test的预测
    def kmean_predict_testdata0(self, test_k):
        train_k = self
        x = test_k.x
        y = test_k.y
        old_total_sse = copy.deepcopy(train_k.total_sse)
        train_k.total_sse = 0
        kmeans_confidence = np.ones((x.shape[0], len(train_k.label))) * 0
        test_k.kmeans_del = 0
        acc1, acc = 0, 0
        total = 0
        index = 0
        min_, sub_min_ = 0, 0

        d = [NAN] * train_k.k
        final_num = [0] * train_k.k
        prob_kmeans = [NAN] * len(set(train_k.y))
        # print(prob_kmeans)
        test_k.kmeans_del = 0
        sum_, min_p = 0, 0
        t, min_index, sub_min_index = 0, 0, 0
        id = NAN  # // 正类在model->label上的索引
        # // 对于测试集（test）中的每一个样本
        for i in range(x.shape[0]):
            # // 下面是初始化
            # // 默认达到第一个质心的距离为最短，d的长度和几个簇有关
            # // d放的是该样本到达每个质心的距离
            d[0] = min_ = distance(train_k.x_c[0], x[i])
            # // 测试样本的预测标签默认为第一个簇
            test_k.c1[i] = train_k.y_c[0]
            test_k.cluster1[i] = 0
            min_index = 0
            # // sum计算样本到每个簇的距离之和
            sum_ = d[0]
            # // 对于从第二个簇开始
            for j in range(1, train_k.k):
                d[j] = distance(train_k.x_c[j], x[i])
                sum_ += d[j]
                if min_ > d[j]:
                    min_ = d[j]
                    test_k.c1[i] = train_k.y_c[j]
                    test_k.cluster1[i] = j
                    min_index = j
            train_k.sse[test_k.cluster1[i]] += min_
            train_k.size[test_k.cluster1[i]] += 1

            if test_k.cluster1[i] == 0:
                sub_min_ = d[1]
                sub_min_index = 1
            else:
                sub_min_ = d[0]
                sub_min_index = 0
            # // 对于每一个簇
            for j in range(train_k.k):
                if sub_min_ > d[j] and j != test_k.cluster1[i]:
                    sub_min_ = d[j]
                    sub_min_index = j
            test_k.u1[i] = 1 - min_ / sub_min_  # //最大减次大作为置信度
            # // // // // // // // // // // // // // // // // // // 根据各个簇的质心距离的组合情况，预测类别
            for t in range(len(train_k.label)):
                prob_kmeans[t] = 0
                for j in range(train_k.k):
                    if train_k.y_c[j] == train_k.label[t]:
                        prob_kmeans[t] += d[j]
                prob_kmeans[t] = (sum_ - prob_kmeans[t]) / sum_
                # // 预测为哪个类的自信度, i表示第几个样本，t表示预测为哪个类
                kmeans_confidence[i][t] = prob_kmeans[t]
                if train_k.label[t] == 1:
                    id = t

            total += 1
            test_k.kmeans_del += test_k.u1[i]
            if test_k.c1[i] == y[i]:
                acc += 1
        test_k.kmeans_confidence = kmeans_confidence
        print("Kmean test accuracy is %f,(%d/%d)" % (double(acc) / x.shape[0], acc, x.shape[0]))
        test_k.first_kmean_predict_testdata0 = double(acc) / x.shape[0]

        # 计算kmeans在正类上的预测概率
        # for (int i = 0; i < test_data.l; i++) {
        #     u3[i] = kmeans_confidence[i][id];
        # }
        for i in range(x.shape[0]):
            test_k.u3[i] = kmeans_confidence[i][id]
        test_k.kmeans_del = test_k.kmeans_del / total;

        total = 0
        acc = 0
        # // 对于每个测试集样本
        for i in range(x.shape[0]):
            train_k.k_c[test_k.cluster1[i]] += 1
            if test_k.u1[i] > test_k.kmeans_del:
                total += 1
                final_num[test_k.cluster1[i]] += 1
                if test_k.c1[i] == y[i]:
                    acc += 1
        print("Kmean confident-test accuracy is %f,(%d/%d)" % (double(acc) / total, acc, total))

        nn = 0
        for j in range(train_k.k):
            acc = 0
            nn = 0
            for i in range(x.shape[0]):
                # // 如果测试样本的簇序号（通过模型预测来的）与该簇一致
                if test_k.cluster1[i] == j:
                    nn += 1
                    if test_k.c1[i] == y[i]:  # //如果测试样本的预测标签和其真实的标签一样，acc+1
                        acc += 1
            print("Cluster %d (Label %d) accuracy is %f,(%d/%d)" % (j, train_k.y_c[j], double(acc) / nn, acc, nn),
                  end="\t")
            train_k.total_sse += train_k.sse[j]
            if train_k.k_c[j] == 0:
                train_k.sse[j] = 0
            else:
                train_k.sse[j] = train_k.sse[j] / train_k.k_c[j]
            print("\tsse[%d]     %lf" % (j, train_k.sse[j]))
        print("\n %lf" % (train_k.total_sse))

        test_k.test_f1 = f1_score(y, test_k.c1, labels=[1])
        print("Testdata f1-value:%f\n" % test_k.test_f1)
        return test_k.c1

    # 二分类，不平衡
    def kmean_split1_1(self, test_k, no):  # 0114根据main_jz.cpp 修改
        #
        train_k = self
        zly_old_old = 0
        zly_old_new = 0

        del_ = 0
        n = 0
        flg = False
        num = [0] * len(train_k.label)
        sum_all = []
        cluster_size = copy.deepcopy(train_k.count[no])
        for k in range(len(train_k.label)):  # //判断属于哪个类别
            sum_ = 0
            num[k] = 0
            for j in range(train_k.x.shape[0]):  # //计算该簇中各个类的样本的“质心”
                if train_k.cluster[j] == no and train_k.y[j] == train_k.label[k]:
                    num[k] += 1
                    sum_ += train_k.x[j]
            # print(sum_)
            sum_all.append(sum_)
        print("Split cluster %d: " % no)
        del_flg = False

        for k in range(len(train_k.label)):
            print("  Label(%d): %d" % (train_k.label[k], num[k]))
            if train_k.label[k] == train_k.y_c[no]:
                if num[k] > 0:

                    train_k.x_c[no] = sum_all[k] / num[k]
                    train_k.parent[no] = no
                    train_k.new_split[no] = 2

                    zly_old_old += 1
                else:
                    del_flg = True

            else:
                if num[k] < 2 and train_k.pos_model[k] != 1:  # TODO  从训练集删除这些大类样本
                    for i in range(train_k.x.shape[0]):
                        if train_k.cluster[i] == no and train_k.pos[i] == 0:
                            train_k.use[i] = 0
                            print("-----------删除第%d个训练样本，标签是:%.0f----------" % (i, train_k.y[i]))
                if (num[k] > 2 or train_k.pos_model[k] == 1) and train_k.k < train_k.maxnum:

                    print("this")
                    flg = True
                    # train_k.x_c[train_k.k]=(sum_all[k] / num[k])
                    if num[k] == 0:
                        train_k.x_c.append(0)
                    else:
                        train_k.x_c.append(sum_all[k] / num[k])
                    train_k.y_c.append(train_k.label[k])
                    train_k.parent[train_k.k] = no
                    train_k.new_split[train_k.k] = 1
                    train_k.k += 1

                    zly_old_new += 1
        if del_flg:  # //如果正确的训练样本数量为0，则把第no个簇（本簇）覆盖掉
            train_k.x_c[no] = copy.deepcopy(train_k.x_c[train_k.k - 1])
            train_k.y_c[no] = copy.deepcopy(train_k.y_c[train_k.k - 1])
            train_k.parent[no] = 1
            train_k.new_split[no] = 1
            train_k.k -= 1
            print("\t%d簇没有正确的训练样本，取消该簇\t" % no)
        print("%d簇被分成了：%d（%d（属于该簇的样本为一个簇）+%d（其他样本簇的数量））\n" % (
            no, zly_old_old + zly_old_new, zly_old_old, zly_old_new));
        return True

    # 删除这个簇
    def kmean_delete_centroid(self, no):
        train_k = self
        f = False
        for i in range(train_k.k):
            if train_k.y_c[i] == train_k.y_c[no] and i != no:
                f = True
        if f == False:
            return False
        train_k.k -= 1
        print("Delete cluster %d;" % no)
        # //对于每个簇，前移一位////
        train_k.k_c[no:] = train_k.k_c[no + 1:]
        train_k.y_c[no:] = train_k.y_c[no + 1:]
        train_k.x_c[no:] = train_k.x_c[no + 1:]
        train_k.size[no:] = train_k.size[no + 1:]
        train_k.pos_model[no:] = train_k.pos_model[no + 1:]

        train_k.count[no:] = train_k.count[no + 1:]
        train_k.old_count[no:] = train_k.old_count[no + 1:]
        train_k.cluster_acc[no:] = train_k.cluster_acc[no + 1:]

        train_k.old_cluster_acc[no:] = train_k.old_cluster_acc[no + 1:]
        train_k.sse[no:] = train_k.sse[no + 1:]
        train_k.old_sse[no:] = train_k.old_sse[no + 1:]

        train_k.parent[no:] = train_k.parent[no + 1:]
        train_k.new_split[no:] = train_k.new_split[no + 1:]
        train_k.count_acc[no:] = train_k.count_acc[no + 1:]
        train_k.old_count_acc[no:] = train_k.old_count_acc[no + 1:]

        # 对于训练样本 下面不知道要不要注释
        train_k.cluster = np.array(train_k.cluster)
        train_k.old_cluster = np.array(train_k.old_cluster)

        train_k.cluster[train_k.cluster == no] = train_k.maxnum - 1
        train_k.old_cluster[train_k.cluster == no] = train_k.maxnum - 1

        train_k.cluster[train_k.cluster > no] -= 1
        train_k.old_cluster[train_k.cluster > no] -= 1
        train_k.cluster = list(train_k.cluster)
        train_k.old_cluster = list(train_k.old_cluster)
        return True

    def kmean_predict0(self):
        min_, max_, sum_ = 0, 0, 0
        t, acc = 0, 0
        y = self.y
        x = self.x
        self.count = [0] * self.k  # 各个簇中训练样本的数量
        self.count_acc = [0] * self.k  # 各个簇中预测正确的训练样本的数量
        self.sse = [0] * self.k  # 簇的密集程度
        self.sse_labeled = [0] * self.k
        self.k_c = [0] * self.k  # 每个簇的样本个数
        self.d = [0] * self.k
        self.size = [0] * self.k  # 每个簇的样本个数
        self.sse_labeled_acc = [0] * self.k

        self.cluster_acc = [0] * self.k
        # zly_new_add
        self.cluster_gmeans = [0] * self.k
        self.cluster = [NAN] * x.shape[0]
        self.u = [NAN] * x.shape[0]
        self.c = [NAN] * x.shape[0]
        self.kmeans_noise_flag = [NAN] * x.shape[0]
        # // 对于训练样本, 初始化质心相关的属性
        for i in range(x.shape[0]):
            # if train_k.use[i] == 0: continue
            # 设置所有的样本都不是噪声
            self.kmeans_noise_flag[i] = False
            # // min：表示初始化，先假设所有的样本都是到第一个质心的距离最短
            # // d[0]: 记录对应的该样本到每个质心的距离
            self.d[0] = min_ = distance(x[i], self.x_c[0])
            sum_ = 1.0 / self.d[0]
            self.c[i] = self.y_c[0]
            self.cluster[i] = 0
            # 以上为初始化

            for j in range(1, self.k):  # 从第二个质心开始
                self.d[j] = distance(self.x_c[j], self.x[i])
                if min_ > self.d[j]:
                    min_ = self.d[j]
                    self.c[i] = self.y_c[j]
                    self.cluster[i] = j
                if self.d[j] == 0:
                    sum_ += 0
                else:
                    sum_ += 1.0 / self.d[j]
            sub_min_ = NAN
            index = NAN
            if self.cluster[i] == 0:
                sub_min_ = self.d[1]
                index = 1
            else:
                sub_min_ = self.d[0]
                index = 0
            # 对于每一个质心
            for j in range(self.k):
                if sub_min_ > self.d[j] and j != self.cluster[i]:
                    sub_min_ = self.d[j]
                    index = j
            self.u[i] = 1 - min_ / sub_min_

            if self.y_c[self.cluster[i]] != self.y[i] and self.u[i] > 0.95:
                self.kmeans_noise_flag[i] = True
            self.count[self.cluster[i]] += 1
            self.k_c[self.cluster[i]] += 1
            self.size[self.cluster[i]] += 1
            self.sse[self.cluster[i]] += min_
            t += 1
            if self.c[i] == self.y[i]:
                acc += 1
        train_acc = double(acc) / t
        print("Kmean train accuracy is {},({}/{})".format(double(acc) / len(self.y), acc, len(self.y)))

        for j in range(self.k):
            self.sse_labeled[j] = self.sse[j]
        # // 计算属于该簇且预测正确的训练样本的SSE
        avg_cluster_acc = 0  # //初始化每个簇的平均准确率为0
        for i in range(self.k):
            # zly_new_add
            curr_true = []
            curr_pred = []
            for j in range(x.shape[0]):
                if self.use[i] == 0: continue
                # 当前样本属于这个簇，且预测正确
                if self.cluster[j] == i and self.y[j] == self.c[j]:
                    self.sse_labeled_acc[i] += distance(self.x_c[i], self.x[j])
                    self.count_acc[i] += 1
                # zly_new_add 如果当前样本输入这个簇
                if self.cluster[j] == i:
                    curr_true.append(copy.deepcopy(self.y[j]))
                    curr_pred.append(copy.deepcopy(self.c[j]))

            if self.count_acc[i] == 0:
                self.sse_labeled_acc[i] = 0
            else:
                self.sse_labeled_acc[i] = self.sse_labeled_acc[i] / self.count_acc[i]
            if self.count[i] == 0:
                self.cluster_acc[i] = 0
            else:
                self.cluster_acc[i] = double(self.count_acc[i]) / self.count[i];
            avg_cluster_acc += self.cluster_acc[i]
            # zly 认为需要修改的地方
            self.cluster_gmeans[i] = G_mean(np.array(curr_true), np.array(curr_pred))
            # train_k.cluster_gmeans[i] = G_mean(np.array(curr_true), np.array(curr_pred))
            # train_k.cluster_gmeans[i] = AUC(np.array(curr_true), np.array(curr_pred))
            print("Cluster %d (Label %d) accuracy is %f,(%d/%d)  Gmean is %f" % (
                i, self.y_c[i], self.cluster_acc[i], self.count_acc[i], self.count[i],
                self.cluster_gmeans[i]), end="\t")
            if self.count[i] == 0:
                print("   Average SSE  %f" % (0))
            else:
                print("   Average SSE  %f" % (self.sse[i] / self.count[i]))
        # zly加上去，计算簇的平均距离
        Dis_avg(self)
        self.avg_cluster_acc = avg_cluster_acc / self.k
        self.train_f1 = f1_score(self.y, self.c, labels=[1])
        print("Train data f1-value:%f\n" % self.train_f1)
        self.train_acc = train_acc
        return self.c


class Kmean_Test:
    def __init__(self, x, y):
        self.x = x  # x
        self.y = y  # y真实的标签
        self.c1 = [NAN] * len(y)  # 预测的标签
        self.kmeans_del = NAN  # kmeans预测样本的置信度阈值
        self.u1 = [NAN] * len(y)  # /测试样本的预测概率
        self.kmeans_confidence = NAN  # 属于每个类的置信度
        self.cluster1 = [NAN] * len(y)  # 属于的簇序号
        self.first_kmean_predict_testdata0 = 0  # // 记录第一次keamns未分裂的ACC
        self.test_acc = NAN
        self.old_test_acc = NAN
        self.select_kmeans = [False] * len(y)  # 是否被选中作为伪标签样本

        # 根据第二次更改新增加的变量
        self.test_f1 = NAN
        self.u3 = [NAN] * len(y)  # 计算kmeans在正类上的预测概率

    def kmean_predict_testdata0(self, train_k):
        test_k = self
        x = test_k.x
        y = test_k.y
        old_total_sse = copy.deepcopy(train_k.total_sse)
        train_k.total_sse = 0
        kmeans_confidence = np.ones((x.shape[0], len(train_k.label))) * 0
        test_k.kmeans_del = 0
        acc1, acc = 0, 0
        total = 0
        index = 0
        min_, sub_min_ = 0, 0

        d = [NAN] * train_k.k
        final_num = [0] * train_k.k
        prob_kmeans = [NAN] * len(set(train_k.y))
        # print(prob_kmeans)
        test_k.kmeans_del = 0
        sum_, min_p = 0, 0
        t, min_index, sub_min_index = 0, 0, 0
        id = NAN  # // 正类在model->label上的索引
        # // 对于测试集（test）中的每一个样本
        for i in range(x.shape[0]):
            # // 下面是初始化
            # // 默认达到第一个质心的距离为最短，d的长度和几个簇有关
            # // d放的是该样本到达每个质心的距离
            d[0] = min_ = distance(train_k.x_c[0], x[i])
            # // 测试样本的预测标签默认为第一个簇
            test_k.c1[i] = train_k.y_c[0]
            test_k.cluster1[i] = 0
            min_index = 0
            # // sum计算样本到每个簇的距离之和
            sum_ = d[0]
            # // 对于从第二个簇开始
            for j in range(1, train_k.k):
                d[j] = distance(train_k.x_c[j], x[i])
                sum_ += d[j]
                if min_ > d[j]:
                    min_ = d[j]
                    test_k.c1[i] = train_k.y_c[j]
                    test_k.cluster1[i] = j
                    min_index = j
            train_k.sse[test_k.cluster1[i]] += min_
            train_k.size[test_k.cluster1[i]] += 1

            if test_k.cluster1[i] == 0:
                sub_min_ = d[1]
                sub_min_index = 1
            else:
                sub_min_ = d[0]
                sub_min_index = 0
            # // 对于每一个簇
            for j in range(train_k.k):
                if sub_min_ > d[j] and j != test_k.cluster1[i]:
                    sub_min_ = d[j]
                    sub_min_index = j
            test_k.u1[i] = 1 - min_ / sub_min_  # //最大减次大作为置信度
            # // // // // // // // // // // // // // // // // // // 根据各个簇的质心距离的组合情况，预测类别
            for t in range(len(train_k.label)):
                prob_kmeans[t] = 0
                for j in range(train_k.k):
                    if train_k.y_c[j] == train_k.label[t]:
                        prob_kmeans[t] += d[j]
                prob_kmeans[t] = (sum_ - prob_kmeans[t]) / sum_
                # // 预测为哪个类的自信度, i表示第几个样本，t表示预测为哪个类
                kmeans_confidence[i][t] = prob_kmeans[t]
                if train_k.label[t] == 1:
                    id = t

            total += 1
            test_k.kmeans_del += test_k.u1[i]
            if test_k.c1[i] == y[i]:
                acc += 1
        test_k.kmeans_confidence = kmeans_confidence
        print("Kmean test accuracy is %f,(%d/%d)" % (double(acc) / x.shape[0], acc, x.shape[0]))
        test_k.first_kmean_predict_testdata0 = double(acc) / x.shape[0]

        # 计算kmeans在正类上的预测概率
        # for (int i = 0; i < test_data.l; i++) {
        #     u3[i] = kmeans_confidence[i][id];
        # }
        for i in range(x.shape[0]):
            test_k.u3[i] = kmeans_confidence[i][id]
        test_k.kmeans_del = test_k.kmeans_del / total;

        total = 0
        acc = 0
        # // 对于每个测试集样本
        for i in range(x.shape[0]):
            train_k.k_c[test_k.cluster1[i]] += 1
            if test_k.u1[i] > test_k.kmeans_del:
                total += 1
                final_num[test_k.cluster1[i]] += 1
                if test_k.c1[i] == y[i]:
                    acc += 1
        print("Kmean confident-test accuracy is %f,(%d/%d)" % (double(acc) / total, acc, total))

        nn = 0
        for j in range(train_k.k):
            acc = 0
            nn = 0
            for i in range(x.shape[0]):
                # // 如果测试样本的簇序号（通过模型预测来的）与该簇一致
                if test_k.cluster1[i] == j:
                    nn += 1
                    if test_k.c1[i] == y[i]:  # //如果测试样本的预测标签和其真实的标签一样，acc+1
                        acc += 1
            print("Cluster %d (Label %d) accuracy is %f,(%d/%d)" % (j, train_k.y_c[j], double(acc) / nn, acc, nn),
                  end="\t")
            train_k.total_sse += train_k.sse[j]
            if train_k.k_c[j] == 0:
                train_k.sse[j] = 0
            else:
                train_k.sse[j] = train_k.sse[j] / train_k.k_c[j]
            print("\tsse[%d]     %lf" % (j, train_k.sse[j]))
        print("\n %lf" % (train_k.total_sse))

        test_k.test_f1 = f1_score(y, test_k.c1, labels=[1])
        print("Testdata f1-value:%f\n" % test_k.test_f1);


#
# 加载数据集
def loadfile(filename):
    x, y = load_svmlight_file(filename)
    x = np.array(x.todense(), dtype=np.float32)
    # x.dtype = np.float16
    return x, np.array(y.astype(np.int))


# 计算两个点的欧式距离
def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


# 计算两个向量的相减
def diminish(x, y):
    ans = []
    for i in range(len(x)):
        ans.append(x[i] - y[i])
    return ans
    # return np.array(x) - np.array(y)


# 实现 double * svm_node
def multiply(x, y):
    ans = []
    for i in range(len(y)):
        ans.append(x * y[i])
    return ans
    # return x * np.array(y)


# numpy 添加元素的方法
def _append_(lis, x):
    lis = lis.tolist()
    lis.append(x)
    lis = np.array(lis)
    return lis


def _extend_(lis, x):
    lis = lis.tolist()
    x = x.tolist()
    lis.extend(x)
    lis = np.array(lis)
    return lis


# 实现两个向量的相加
def add1(x, y):
    ans = []
    for i in range(len(x)):
        ans.append(x[i] + y[i])
    return ans


# 初始化质心
def k_center(train_k):
    x, y = train_k.x, train_k.y
    # print("``")
    # print(len(y))
    # print(range(len(K_data.label)),range(len(y)))
    for i in range(train_k.k):
        train_k.k_c[i] = 0
        train_k.y_c[i] = train_k.label[i]
    for i in range(train_k.k):  # 对于每一个簇
        if train_k.use[i] == 0: continue
        sum = 0
        sum_num = 0
        for j in range(len(y)):  # 对于每一个样本
            if y[j] == train_k.label[i]:  # 如果样本的标签和簇的标签一样的话

                train_k.k_c[i] += 1  # 簇中的样本个数+1
                train_k.count[i] += 1  # 簇中的训练样本个数+1
                train_k.cluster[j] = i  # 训练样本所属的簇序号
                train_k.c[j] = y[j]
                sum += x[j]
                sum_num += 1
        train_k.x_c[i] = sum / sum_num
        # print(K_data.x_c[i])
    print("train class num :")
    for i in range(train_k.k):  # 对于每一个簇
        print("label({}):{}\t".format(train_k.label[i], train_k.k_c[i]), end="\t")
        x_ls = x[y == train_k.label[i]]
        # print(x_ls.shape)
        sse_this = 0
        for x2 in x_ls:
            # print(np.sqrt(np.sum(np.square(x2 - K_data.x_c[i]))))
            sse_this += distance(x2, train_k.x_c[i]) ** 2
        train_k.sse[i] = sse_this / x_ls.shape[0]
        print("sse:", train_k.sse[i])
        train_k.count_acc[i] = train_k.count[i]
        train_k.cluster_acc[i] = 1


def kmean_predict0_no_show(train_k):
    min_, max_, sum_ = 0, 0, 0
    t, acc = 0, 0
    y = train_k.y
    x = train_k.x
    train_k.count = [0] * train_k.k  # 各个簇中训练样本的数量
    train_k.count_acc = [0] * train_k.k  # 各个簇中预测正确的训练样本的数量
    train_k.sse = [0] * train_k.k  # 簇的密集程度
    train_k.sse_labeled = [0] * train_k.k
    train_k.k_c = [0] * train_k.k  # 每个簇的样本个数
    train_k.d = [0] * train_k.k
    train_k.size = [0] * train_k.k  # 每个簇的样本个数
    train_k.sse_labeled_acc = [0] * train_k.k

    train_k.cluster_acc = [0] * train_k.k
    # zly_new_add
    train_k.cluster_gmeans = [0] * train_k.k
    train_k.cluster = [NAN] * x.shape[0]
    train_k.u = [NAN] * x.shape[0]
    # // 对于训练样本, 初始化质心相关的属性
    for i in range(x.shape[0]):
        if train_k.use[i] == 0: continue
        # 设置所有的样本都不是噪声
        train_k.kmeans_noise_flag[i] = False
        # // min：表示初始化，先假设所有的样本都是到第一个质心的距离最短
        # // d[0]: 记录对应的该样本到每个质心的距离
        train_k.d[0] = min_ = distance(x[i], train_k.x_c[0])
        sum_ = 1.0 / train_k.d[0]
        train_k.c[i] = train_k.y_c[0]
        train_k.cluster[i] = 0
        # 以上为初始化

        for j in range(1, train_k.k):  # 从第二个质心开始
            train_k.d[j] = distance(train_k.x_c[j], train_k.x[i])
            if min_ > train_k.d[j]:
                min_ = train_k.d[j]
                train_k.c[i] = train_k.y_c[j]
                train_k.cluster[i] = j
            if train_k.d[j] == 0:
                sum_ += 0
            else:
                sum_ += 1.0 / train_k.d[j]
        sub_min_ = NAN
        index = NAN
        if train_k.cluster[i] == 0:
            sub_min_ = train_k.d[1]
            index = 1
        else:
            sub_min_ = train_k.d[0]
            index = 0
        # 对于每一个质心
        for j in range(train_k.k):
            if sub_min_ > train_k.d[j] and j != train_k.cluster[i]:
                sub_min_ = train_k.d[j]
                index = j
        train_k.u[i] = 1 - min_ / sub_min_

        if train_k.y_c[train_k.cluster[i]] != train_k.y[i] and train_k.u[i] > 0.95:
            train_k.kmeans_noise_flag[i] = True
        train_k.count[train_k.cluster[i]] += 1
        train_k.k_c[train_k.cluster[i]] += 1
        train_k.size[train_k.cluster[i]] += 1
        train_k.sse[train_k.cluster[i]] += min_
        t += 1
        if train_k.c[i] == train_k.y[i]:
            acc += 1
    train_acc = double(acc) / t

    for j in range(train_k.k):
        train_k.sse_labeled[j] = train_k.sse[j]
    # // 计算属于该簇且预测正确的训练样本的SSE
    avg_cluster_acc = 0  # //初始化每个簇的平均准确率为0
    for i in range(train_k.k):
        # zly_new_add
        curr_true = []
        curr_pred = []
        for j in range(x.shape[0]):
            if train_k.use[i] == 0: continue
            # 当前样本属于这个簇，且预测正确
            if train_k.cluster[j] == i and train_k.y[j] == train_k.c[j]:
                train_k.sse_labeled_acc[i] += distance(train_k.x_c[i], train_k.x[j])
                train_k.count_acc[i] += 1
            # zly_new_add 如果当前样本输入这个簇
            if train_k.cluster[j] == i:
                curr_true.append(copy.deepcopy(train_k.y[j]))
                curr_pred.append(copy.deepcopy(train_k.c[j]))

        if train_k.count_acc[i] == 0:
            train_k.sse_labeled_acc[i] = 0
        else:
            train_k.sse_labeled_acc[i] = train_k.sse_labeled_acc[i] / train_k.count_acc[i]
        if train_k.count[i] == 0:
            train_k.cluster_acc[i] = 0
        else:
            train_k.cluster_acc[i] = double(train_k.count_acc[i]) / train_k.count[i];
        avg_cluster_acc += train_k.cluster_acc[i]
        # zly 认为需要修改的地方
        train_k.cluster_gmeans[i] = G_mean(np.array(curr_true), np.array(curr_pred), labels=[1])
        # train_k.cluster_gmeans[i] = G_mean(np.array(curr_true), np.array(curr_pred))
        # train_k.cluster_gmeans[i] = AUC(np.array(curr_true), np.array(curr_pred))

    # zly加上去，计算簇的平均距离
    Dis_avg(train_k)
    train_k.avg_cluster_acc = avg_cluster_acc / train_k.k
    train_k.train_f1 = f1_score(train_k.y, train_k.c, labels=[1])
    train_k.train_acc = train_acc


def kmean_predict0(train_k):
    min_, max_, sum_ = 0, 0, 0
    t, acc = 0, 0
    y = train_k.y
    x = train_k.x
    train_k.count = [0] * train_k.k  # 各个簇中训练样本的数量
    train_k.count_acc = [0] * train_k.k  # 各个簇中预测正确的训练样本的数量
    train_k.sse = [0] * train_k.k  # 簇的密集程度
    train_k.sse_labeled = [0] * train_k.k
    train_k.k_c = [0] * train_k.k  # 每个簇的样本个数
    train_k.d = [0] * train_k.k
    train_k.size = [0] * train_k.k  # 每个簇的样本个数
    train_k.sse_labeled_acc = [0] * train_k.k

    train_k.cluster_acc = [0] * train_k.k
    # zly_new_add
    train_k.cluster_gmeans = [0] * train_k.k
    train_k.cluster = [NAN] * x.shape[0]
    train_k.u = [NAN] * x.shape[0]
    train_k.c = [NAN] * x.shape[0]
    train_k.kmeans_noise_flag = [NAN] * x.shape[0]
    # // 对于训练样本, 初始化质心相关的属性
    for i in range(x.shape[0]):
        # if train_k.use[i] == 0: continue
        # 设置所有的样本都不是噪声
        train_k.kmeans_noise_flag[i] = False
        # // min：表示初始化，先假设所有的样本都是到第一个质心的距离最短
        # // d[0]: 记录对应的该样本到每个质心的距离
        train_k.d[0] = min_ = distance(x[i], train_k.x_c[0])
        sum_ = 1.0 / train_k.d[0]
        train_k.c[i] = train_k.y_c[0]
        train_k.cluster[i] = 0
        # 以上为初始化

        for j in range(1, train_k.k):  # 从第二个质心开始
            train_k.d[j] = distance(train_k.x_c[j], train_k.x[i])
            if min_ > train_k.d[j]:
                min_ = train_k.d[j]
                train_k.c[i] = train_k.y_c[j]
                train_k.cluster[i] = j
            if train_k.d[j] == 0:
                sum_ += 0
            else:
                sum_ += 1.0 / train_k.d[j]
        sub_min_ = NAN
        index = NAN
        if train_k.cluster[i] == 0:
            sub_min_ = train_k.d[1]
            index = 1
        else:
            sub_min_ = train_k.d[0]
            index = 0
        # 对于每一个质心
        for j in range(train_k.k):
            if sub_min_ > train_k.d[j] and j != train_k.cluster[i]:
                sub_min_ = train_k.d[j]
                index = j
        train_k.u[i] = 1 - min_ / sub_min_

        if train_k.y_c[train_k.cluster[i]] != train_k.y[i] and train_k.u[i] > 0.95:
            train_k.kmeans_noise_flag[i] = True
        train_k.count[train_k.cluster[i]] += 1
        train_k.k_c[train_k.cluster[i]] += 1
        train_k.size[train_k.cluster[i]] += 1
        train_k.sse[train_k.cluster[i]] += min_
        t += 1
        if train_k.c[i] == train_k.y[i]:
            acc += 1
    train_acc = double(acc) / t
    print("Kmean train accuracy is {},({}/{})".format(double(acc) / len(train_k.y), acc, len(train_k.y)))

    for j in range(train_k.k):
        train_k.sse_labeled[j] = train_k.sse[j]
    # // 计算属于该簇且预测正确的训练样本的SSE
    avg_cluster_acc = 0  # //初始化每个簇的平均准确率为0
    for i in range(train_k.k):
        # zly_new_add
        curr_true = []
        curr_pred = []
        for j in range(x.shape[0]):
            if train_k.use[i] == 0: continue
            # 当前样本属于这个簇，且预测正确
            if train_k.cluster[j] == i and train_k.y[j] == train_k.c[j]:
                train_k.sse_labeled_acc[i] += distance(train_k.x_c[i], train_k.x[j])
                train_k.count_acc[i] += 1
            # zly_new_add 如果当前样本输入这个簇
            if train_k.cluster[j] == i:
                curr_true.append(copy.deepcopy(train_k.y[j]))
                curr_pred.append(copy.deepcopy(train_k.c[j]))

        if train_k.count_acc[i] == 0:
            train_k.sse_labeled_acc[i] = 0
        else:
            train_k.sse_labeled_acc[i] = train_k.sse_labeled_acc[i] / train_k.count_acc[i]
        if train_k.count[i] == 0:
            train_k.cluster_acc[i] = 0
        else:
            train_k.cluster_acc[i] = double(train_k.count_acc[i]) / train_k.count[i];
        avg_cluster_acc += train_k.cluster_acc[i]
        # zly 认为需要修改的地方
        train_k.cluster_gmeans[i] = G_mean(np.array(curr_true), np.array(curr_pred))
        # train_k.cluster_gmeans[i] = G_mean(np.array(curr_true), np.array(curr_pred))
        # train_k.cluster_gmeans[i] = AUC(np.array(curr_true), np.array(curr_pred))
        print("Cluster %d (Label %d) accuracy is %f,(%d/%d)  Gmean is %f" % (
            i, train_k.y_c[i], train_k.cluster_acc[i], train_k.count_acc[i], train_k.count[i],
            train_k.cluster_gmeans[i]), end="\t")
        if train_k.count[i] == 0:
            print("   Average SSE  %f" % (0))
        else:
            print("   Average SSE  %f" % (train_k.sse[i] / train_k.count[i]))
    # zly加上去，计算簇的平均距离
    Dis_avg(train_k)
    train_k.avg_cluster_acc = avg_cluster_acc / train_k.k
    train_k.train_f1 = f1_score(train_k.y, train_k.c, labels=[1])
    print("Train data f1-value:%f\n" % train_k.train_f1)
    train_k.train_acc = train_acc


def Dis_avg(train_k):
    x_c = train_k.x_c
    dis_ = 0
    for i in range(len(x_c)):
        for j in range(len(x_c)):
            if i != j:
                dis_ += dist(x_c[i], x_c[j])
    dis_ = dis_ * 1 / (train_k.k * (train_k.k - 1))
    train_k.Dis = dis_


# 计算两个向量的欧式距离
def dist(a, b):
    return np.sqrt(sum((a - b) ** 2))


def kmean_predict_testdata0(train_k, test_k):
    x = test_k.x
    y = test_k.y
    old_total_sse = copy.deepcopy(train_k.total_sse)
    train_k.total_sse = 0
    kmeans_confidence = np.ones((x.shape[0], len(train_k.label))) * 0
    test_k.kmeans_del = 0
    acc1, acc = 0, 0
    total = 0
    index = 0
    min_, sub_min_ = 0, 0

    d = [NAN] * train_k.k
    final_num = [0] * train_k.k
    prob_kmeans = [NAN] * len(set(train_k.y))
    # print(prob_kmeans)
    test_k.kmeans_del = 0
    sum_, min_p = 0, 0
    t, min_index, sub_min_index = 0, 0, 0
    id = NAN  # // 正类在model->label上的索引
    # // 对于测试集（test）中的每一个样本
    for i in range(x.shape[0]):
        # // 下面是初始化
        # // 默认达到第一个质心的距离为最短，d的长度和几个簇有关
        # // d放的是该样本到达每个质心的距离
        d[0] = min_ = distance(train_k.x_c[0], x[i])
        # // 测试样本的预测标签默认为第一个簇
        test_k.c1[i] = train_k.y_c[0]
        test_k.cluster1[i] = 0
        min_index = 0
        # // sum计算样本到每个簇的距离之和
        sum_ = d[0]
        # // 对于从第二个簇开始
        for j in range(1, train_k.k):
            d[j] = distance(train_k.x_c[j], x[i])
            sum_ += d[j]
            if min_ > d[j]:
                min_ = d[j]
                test_k.c1[i] = train_k.y_c[j]
                test_k.cluster1[i] = j
                min_index = j
        train_k.sse[test_k.cluster1[i]] += min_
        train_k.size[test_k.cluster1[i]] += 1

        if test_k.cluster1[i] == 0:
            sub_min_ = d[1]
            sub_min_index = 1
        else:
            sub_min_ = d[0]
            sub_min_index = 0
        # // 对于每一个簇
        for j in range(train_k.k):
            if sub_min_ > d[j] and j != test_k.cluster1[i]:
                sub_min_ = d[j]
                sub_min_index = j
        test_k.u1[i] = 1 - min_ / sub_min_  # //最大减次大作为置信度
        # // // // // // // // // // // // // // // // // // // 根据各个簇的质心距离的组合情况，预测类别
        for t in range(len(train_k.label)):
            prob_kmeans[t] = 0
            for j in range(train_k.k):
                if train_k.y_c[j] == train_k.label[t]:
                    prob_kmeans[t] += d[j]
            prob_kmeans[t] = (sum_ - prob_kmeans[t]) / sum_
            # // 预测为哪个类的自信度, i表示第几个样本，t表示预测为哪个类
            kmeans_confidence[i][t] = prob_kmeans[t]
            if train_k.label[t] == 1:
                id = t

        total += 1
        test_k.kmeans_del += test_k.u1[i]
        if test_k.c1[i] == y[i]:
            acc += 1
    test_k.kmeans_confidence = kmeans_confidence
    print("Kmean test accuracy is %f,(%d/%d)" % (double(acc) / x.shape[0], acc, x.shape[0]))
    test_k.first_kmean_predict_testdata0 = double(acc) / x.shape[0]

    # 计算kmeans在正类上的预测概率
    # for (int i = 0; i < test_data.l; i++) {
    #     u3[i] = kmeans_confidence[i][id];
    # }
    for i in range(x.shape[0]):
        test_k.u3[i] = kmeans_confidence[i][id]
    test_k.kmeans_del = test_k.kmeans_del / total;

    total = 0
    acc = 0
    # // 对于每个测试集样本
    for i in range(x.shape[0]):
        train_k.k_c[test_k.cluster1[i]] += 1
        if test_k.u1[i] > test_k.kmeans_del:
            total += 1
            final_num[test_k.cluster1[i]] += 1
            if test_k.c1[i] == y[i]:
                acc += 1
    print("Kmean confident-test accuracy is %f,(%d/%d)" % (double(acc) / total, acc, total))

    nn = 0
    for j in range(train_k.k):
        acc = 0
        nn = 0
        for i in range(x.shape[0]):
            # // 如果测试样本的簇序号（通过模型预测来的）与该簇一致
            if test_k.cluster1[i] == j:
                nn += 1
                if test_k.c1[i] == y[i]:  # //如果测试样本的预测标签和其真实的标签一样，acc+1
                    acc += 1
        print("Cluster %d (Label %d) accuracy is %f,(%d/%d)" % (j, train_k.y_c[j], double(acc) / nn, acc, nn),
              end="\t")
        train_k.total_sse += train_k.sse[j]
        if train_k.k_c[j] == 0:
            train_k.sse[j] = 0
        else:
            train_k.sse[j] = train_k.sse[j] / train_k.k_c[j]
        print("\tsse[%d]     %lf" % (j, train_k.sse[j]))
    print("\n %lf" % (train_k.total_sse))

    test_k.test_f1 = f1_score(y, test_k.c1, labels=[1])
    print("Testdata f1-value:%f\n" % test_k.test_f1);


def kmean_predict_testdata0_no_show(train_k, test_k):
    x = test_k.x
    y = test_k.y
    old_total_sse = copy.deepcopy(train_k.total_sse)
    train_k.total_sse = 0
    kmeans_confidence = np.ones((x.shape[0], len(train_k.label))) * 0
    test_k.kmeans_del = 0
    acc1, acc = 0, 0
    total = 0
    index = 0
    min_, sub_min_ = 0, 0

    d = [NAN] * train_k.k
    final_num = [0] * train_k.k
    prob_kmeans = [NAN] * len(set(train_k.y))
    # print(prob_kmeans)
    test_k.kmeans_del = 0
    sum_, min_p = 0, 0
    t, min_index, sub_min_index = 0, 0, 0
    id = NAN  # // 正类在model->label上的索引
    # // 对于测试集（test）中的每一个样本
    for i in range(x.shape[0]):
        # // 下面是初始化
        # // 默认达到第一个质心的距离为最短，d的长度和几个簇有关
        # // d放的是该样本到达每个质心的距离
        d[0] = min_ = distance(train_k.x_c[0], x[i])
        # // 测试样本的预测标签默认为第一个簇
        test_k.c1[i] = train_k.y_c[0]
        test_k.cluster1[i] = 0
        min_index = 0
        # // sum计算样本到每个簇的距离之和
        sum_ = d[0]
        # // 对于从第二个簇开始
        for j in range(1, train_k.k):
            d[j] = distance(train_k.x_c[j], x[i])
            sum_ += d[j]
            if min_ > d[j]:
                min_ = d[j]
                test_k.c1[i] = train_k.y_c[j]
                test_k.cluster1[i] = j
                min_index = j
        train_k.sse[test_k.cluster1[i]] += min_
        train_k.size[test_k.cluster1[i]] += 1

        if test_k.cluster1[i] == 0:
            sub_min_ = d[1]
            sub_min_index = 1
        else:
            sub_min_ = d[0]
            sub_min_index = 0
        # // 对于每一个簇
        for j in range(train_k.k):
            if sub_min_ > d[j] and j != test_k.cluster1[i]:
                sub_min_ = d[j]
                sub_min_index = j
        test_k.u1[i] = 1 - min_ / sub_min_  # //最大减次大作为置信度
        # // // // // // // // // // // // // // // // // // // 根据各个簇的质心距离的组合情况，预测类别
        for t in range(len(train_k.label)):
            prob_kmeans[t] = 0
            for j in range(train_k.k):
                if train_k.y_c[j] == train_k.label[t]:
                    prob_kmeans[t] += d[j]
            prob_kmeans[t] = (sum_ - prob_kmeans[t]) / sum_
            # // 预测为哪个类的自信度, i表示第几个样本，t表示预测为哪个类
            kmeans_confidence[i][t] = prob_kmeans[t]
            if train_k.label[t] == 1:
                id = t

        total += 1
        test_k.kmeans_del += test_k.u1[i]
        if test_k.c1[i] == y[i]:
            acc += 1
    test_k.kmeans_confidence = kmeans_confidence

    test_k.first_kmean_predict_testdata0 = double(acc) / x.shape[0]

    # 计算kmeans在正类上的预测概率
    # for (int i = 0; i < test_data.l; i++) {
    #     u3[i] = kmeans_confidence[i][id];
    # }
    for i in range(x.shape[0]):
        test_k.u3[i] = kmeans_confidence[i][id]
    test_k.kmeans_del = test_k.kmeans_del / total;

    total = 0
    acc = 0
    # // 对于每个测试集样本
    for i in range(x.shape[0]):
        train_k.k_c[test_k.cluster1[i]] += 1
        if test_k.u1[i] > test_k.kmeans_del:
            total += 1
            final_num[test_k.cluster1[i]] += 1
            if test_k.c1[i] == y[i]:
                acc += 1

    nn = 0
    for j in range(train_k.k):
        acc = 0
        nn = 0
        for i in range(x.shape[0]):
            # // 如果测试样本的簇序号（通过模型预测来的）与该簇一致
            if test_k.cluster1[i] == j:
                nn += 1
                if test_k.c1[i] == y[i]:  # //如果测试样本的预测标签和其真实的标签一样，acc+1
                    acc += 1

        train_k.total_sse += train_k.sse[j]
        if train_k.k_c[j] == 0:
            train_k.sse[j] = 0
        else:
            train_k.sse[j] = train_k.sse[j] / train_k.k_c[j]

    test_k.test_f1 = f1_score(y, test_k.c1, labels=[1])


def kmean_split1(train_k, test_k, no):
    #
    zly_old_old = 0
    zly_old_new = 0

    if train_k.count[no] <= 1:
        kmean_delete_centroid(train_k, no)
        train_k.old_k -= 1
        return True
    del_ = 0
    n = 0
    flg = False
    num = [0] * len(train_k.label)
    sum_all = []
    cluster_size = copy.deepcopy(train_k.count[no])
    for k in range(len(train_k.label)):  # //判断属于哪个类别
        sum_ = 0
        num[k] = 0
        for j in range(train_k.x.shape[0]):  # //计算该簇中各个类的样本的“质心”
            if train_k.cluster[j] == no and train_k.y[j] == train_k.label[k]:
                num[k] += 1
                sum_ += train_k.x[j]
        # print(sum_)
        sum_all.append(sum_)
    print("Split cluster %d: " % no)
    del_flg = False

    for k in range(len(train_k.label)):
        print("  Label-%d: %d" % (train_k.label[k], num[k]))
        if train_k.label[k] == train_k.y_c[no]:
            if num[k] > 0:

                train_k.x_c[no] = sum_all[k] / num[k]
                train_k.parent[no] = no
                train_k.new_split[no] = 2
                print("预测正确的样本分为一个簇(no:%d->%d,num[label:%d]:%d)\n" % (no, no, train_k.label[k], num[k]))
                zly_old_old += 1
            else:
                del_flg = True
        elif (num[k] > 2 or double(num[k]) / cluster_size > 0.05) and train_k.k < train_k.maxnum:
            flg = True
            # train_k.x_c[train_k.k]=(sum_all[k] / num[k])
            train_k.x_c.append(sum_all[k] / num[k])
            train_k.y_c.append(train_k.label[k])
            train_k.parent[train_k.k] = no
            train_k.new_split[train_k.k] = 1
            train_k.k += 1
            print(
                "预测不正确的样本分为一个簇(no:%d->%d,num[label:%d]:%d)\n" % (no, train_k.k, train_k.label[k], num[k]))
            zly_old_new += 1
    if del_flg:  # //如果正确的训练样本数量为0，则把第no个簇（本簇）覆盖掉
        train_k.x_c[no] = copy.deepcopy(train_k.x_c[train_k.k - 1])
        train_k.y_c[no] = copy.deepcopy(train_k.y_c[train_k.k - 1])
        train_k.parent[no] = 1
        train_k.new_split[no] = 1
        train_k.k -= 1
        print("\t%d簇没有正确的训练样本，取消该簇\t" % no)
    print("%d簇被分成了：%d（%d（属于该簇的样本为一个簇）+%d（其他样本簇的数量））\n" % (
        no, zly_old_old + zly_old_new, zly_old_old, zly_old_new));
    return True


def kmean_split1_1(train_k, test_k, no):  # 0114根据main_jz.cpp 修改
    #
    zly_old_old = 0
    zly_old_new = 0

    del_ = 0
    n = 0
    flg = False
    num = [0] * len(train_k.label)
    sum_all = []
    cluster_size = copy.deepcopy(train_k.count[no])
    for k in range(len(train_k.label)):  # //判断属于哪个类别
        sum_ = 0
        num[k] = 0
        for j in range(train_k.x.shape[0]):  # //计算该簇中各个类的样本的“质心”
            if train_k.cluster[j] == no and train_k.y[j] == train_k.label[k]:
                num[k] += 1
                sum_ += train_k.x[j]
        # print(sum_)
        sum_all.append(sum_)
    print("Split cluster %d: " % no)
    del_flg = False

    for k in range(len(train_k.label)):
        print("  Label(%d): %d" % (train_k.label[k], num[k]))
        if train_k.label[k] == train_k.y_c[no]:
            if num[k] > 0:

                train_k.x_c[no] = sum_all[k] / num[k]
                train_k.parent[no] = no
                train_k.new_split[no] = 2

                zly_old_old += 1
            else:
                del_flg = True

        else:
            if num[k] < 2 and train_k.pos_model[k] != 1:  # TODO  从训练集删除这些大类样本
                for i in range(train_k.x.shape[0]):
                    if train_k.cluster[i] == no and train_k.pos[i] == 0:
                        train_k.use[i] = 0
                        print("-----------删除第%d个训练样本，标签是:%.0f----------" % (i, train_k.y[i]))
            if (num[k] > 2 or train_k.pos_model[k] == 1) and train_k.k < train_k.maxnum:

                print("this")
                flg = True
                # train_k.x_c[train_k.k]=(sum_all[k] / num[k])
                if num[k] == 0:
                    train_k.x_c.append(0)
                else:
                    train_k.x_c.append(sum_all[k] / num[k])
                train_k.y_c.append(train_k.label[k])
                train_k.parent[train_k.k] = no
                train_k.new_split[train_k.k] = 1
                train_k.k += 1

                zly_old_new += 1
    if del_flg:  # //如果正确的训练样本数量为0，则把第no个簇（本簇）覆盖掉
        train_k.x_c[no] = copy.deepcopy(train_k.x_c[train_k.k - 1])
        train_k.y_c[no] = copy.deepcopy(train_k.y_c[train_k.k - 1])
        train_k.parent[no] = 1
        train_k.new_split[no] = 1
        train_k.k -= 1
        print("\t%d簇没有正确的训练样本，取消该簇\t" % no)
    print("%d簇被分成了：%d（%d（属于该簇的样本为一个簇）+%d（其他样本簇的数量））\n" % (
        no, zly_old_old + zly_old_new, zly_old_old, zly_old_new));
    return True


# 删除这个簇
# tips,未来的师弟师妹们注意，这边删除簇的时候，可能会出现bug，根据需要部分属性，比如train_k.parent[no:] = train_k.parent[no + 1:]改成
# train_k.parent[no:] = train_k.parent[no + 1:] ，train_k.parent.append(初始值)
def kmean_delete_centroid(train_k, no):
    f = False
    for i in range(train_k.k):
        if train_k.y_c[i] == train_k.y_c[no] and i != no:
            f = True
    if f == False:
        return False
    train_k.k -= 1
    print("Delete cluster %d;" % no)
    # //对于每个簇，前移一位////
    train_k.k_c[no:] = train_k.k_c[no + 1:]
    train_k.y_c[no:] = train_k.y_c[no + 1:]
    train_k.x_c[no:] = train_k.x_c[no + 1:]
    train_k.size[no:] = train_k.size[no + 1:]
    train_k.pos_model[no:] = train_k.pos_model[no + 1:]

    train_k.count[no:] = train_k.count[no + 1:]
    train_k.old_count[no:] = train_k.old_count[no + 1:]
    train_k.cluster_acc[no:] = train_k.cluster_acc[no + 1:]

    train_k.old_cluster_acc[no:] = train_k.old_cluster_acc[no + 1:]
    train_k.sse[no:] = train_k.sse[no + 1:]
    train_k.old_sse[no:] = train_k.old_sse[no + 1:]

    train_k.parent[no:] = train_k.parent[no + 1:]
    train_k.parent.append(-1)
    train_k.new_split[no:] = train_k.new_split[no + 1:]
    train_k.new_split.append(0)
    train_k.count_acc[no:] = train_k.count_acc[no + 1:]

    train_k.old_count_acc[no:] = train_k.old_count_acc[no + 1:]

    # 对于训练样本 下面不知道要不要注释
    train_k.cluster = np.array(train_k.cluster)
    train_k.old_cluster = np.array(train_k.old_cluster)

    train_k.cluster[train_k.cluster == no] = train_k.maxnum - 1
    train_k.old_cluster[train_k.cluster == no] = train_k.maxnum - 1

    train_k.cluster[train_k.cluster > no] -= 1
    train_k.old_cluster[train_k.cluster > no] -= 1
    train_k.cluster = list(train_k.cluster)
    train_k.old_cluster = list(train_k.old_cluster)
    return True


# 删除这个簇中所有的多数类样本:
def kmean_delete_minority_items(train_k, no):
    x = train_k.x
    y = train_k.y
    c = train_k.c
    pos = train_k.pos
    u = train_k.u
    cluster = train_k.cluster

    del_ls = []
    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1:
            del_ls.append(i)
            # print("训练样本 %d 删除" % (i))
    print(del_ls)
    x = [x[i] for i in range(len(x)) if (i not in del_ls)]
    y = [y[i] for i in range(len(y)) if (i not in del_ls)]
    cluster = [cluster[i] for i in range(len(cluster)) if (i not in del_ls)]
    c = [c[i] for i in range(len(c)) if (i not in del_ls)]
    pos = [pos[i] for i in range(len(pos)) if (i not in del_ls)]
    u = [u[i] for i in range(len(u)) if (i not in del_ls)]
    train_k.x = np.array(x)
    train_k.y = np.array(y)
    train_k.cluster = cluster
    train_k.c = c
    train_k.pos = pos
    train_k.u = u


# 删除这个簇和这个簇中所有的样本
def kmean_delete_cluster_And_items(train_k, no):
    f = False
    for i in range(train_k.k):
        if train_k.y_c[i] == train_k.y_c[no] and i != no:
            f = True
    if f == False:
        return False
    train_k.k -= 1
    print("Delete cluster %d " % no)
    # 在删除该簇前，先删除该簇内的所有训练样本(少数类不删除)
    x = train_k.x
    y = train_k.y
    c = train_k.c
    pos = train_k.pos
    u = train_k.u
    cluster = train_k.cluster
    i = 0
    del_ls = []

    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1:
            # if train_k.cluster[i] == no :
            del_ls.append(i)
            # print("训练样本 %d 删除" % (i))
    del i
    print("本轮删除的多数类样本个数{}".format(len(del_ls)))
    print(del_ls)
    x = [x[i] for i in range(len(x)) if (i not in del_ls)]
    y = [y[i] for i in range(len(y)) if (i not in del_ls)]
    cluster = [cluster[i] for i in range(len(cluster)) if (i not in del_ls)]
    c = [c[i] for i in range(len(c)) if (i not in del_ls)]
    pos = [pos[i] for i in range(len(pos)) if (i not in del_ls)]
    u = [u[i] for i in range(len(u)) if (i not in del_ls)]
    train_k.x = np.array(x)
    train_k.y = np.array(y)
    train_k.cluster = cluster
    train_k.c = c
    train_k.pos = pos
    train_k.u = u
    #
    # //对于每个簇，前移一位////
    # 后加的

    train_k.cluster_gmeans[no:] = train_k.cluster_gmeans[no + 1:]
    # 过去的
    train_k.k_c[no:] = train_k.k_c[no + 1:]
    train_k.y_c[no:] = train_k.y_c[no + 1:]
    train_k.x_c[no:] = train_k.x_c[no + 1:]
    train_k.size[no:] = train_k.size[no + 1:]
    # train_k.pos_model[no:] = train_k.pos_model[no + 1:]

    train_k.count[no:] = train_k.count[no + 1:]
    train_k.old_count[no:] = train_k.old_count[no + 1:]
    train_k.cluster_acc[no:] = train_k.cluster_acc[no + 1:]

    train_k.old_cluster_acc[no:] = train_k.old_cluster_acc[no + 1:]
    train_k.sse[no:] = train_k.sse[no + 1:]
    train_k.old_sse[no:] = train_k.old_sse[no + 1:]

    train_k.parent[no:] = train_k.parent[no + 1:]
    train_k.new_split[no:] = train_k.new_split[no + 1:]
    train_k.count_acc[no:] = train_k.count_acc[no + 1:]
    train_k.old_count_acc[no:] = train_k.old_count_acc[no + 1:]
    # 对于训练样本 下面不知道要不要注释
    #     train_k.cluster = np.array(train_k.cluster)
    #     train_k.old_cluster = np.array(train_k.old_cluster)
    #
    #     train_k.cluster[train_k.cluster == no] = train_k.maxnum - 1
    #     train_k.old_cluster[train_k.cluster == no] = train_k.maxnum - 1
    #
    #     train_k.cluster[train_k.cluster > no] -= 1
    #     train_k.old_cluster[train_k.cluster > no] -= 1
    #     train_k.cluster = list(train_k.cluster)
    #     train_k.old_cluster = list(train_k.old_cluster)

    return True


# 改变这个簇中，位于边缘的点，直接变为1;
def kmean_change_cluster_Neg_furthest_items(train_k, no, percentage=0.1):
    percentage = 1 - percentage
    print("change cluster %d furthest Negtivate items" % no)

    x = train_k.x
    y = train_k.y
    c = train_k.c
    pos = train_k.pos
    u = train_k.u
    cluster = train_k.cluster

    # 保存这个簇中样本到质心的距离
    d_neg = []
    # 当前簇的质心
    curr_Center = train_k.x_c[no]
    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1:
            # if train_k.cluster[i] == no :

            d_neg.append(distance(x[i], curr_Center))
            # print("训练样本 %d 删除" % (i))
    d_neg.sort()
    del_line = d_neg[int(len(d_neg) * percentage)]
    print("距离界限:{}".format(del_line))
    # 开始删除最靠近边缘的10%样本
    #####
    del_ls = []
    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1 and distance(x[i], curr_Center) > del_line:
            # if train_k.cluster[i] == no :
            del_ls.append(i)
            train_k.y[i] = 1
            # print("训练样本 %d 删除" % (i))
    print("本轮{}修改的多数类样本个数{}".format(no, len(del_ls)))
    print(del_ls)


# 删除这个簇中距离中心最远的样本(只删除多数类)
def kmean_delete_cluster_Neg_furthest_items(train_k, no, percentage=0.9):
    percentage = 1 - percentage
    print("Delete cluster %d furthest Negtivate items" % no)

    x = train_k.x
    y = train_k.y
    c = train_k.c
    pos = train_k.pos
    u = train_k.u
    cluster = train_k.cluster

    # 保存这个簇中样本到质心的距离
    d_neg = []
    # 当前簇的质心
    curr_Center = train_k.x_c[no]
    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1:
            # if train_k.cluster[i] == no :

            d_neg.append(distance(x[i], curr_Center))
            # print("训练样本 %d 删除" % (i))
    d_neg.sort()
    del_line = d_neg[int(len(d_neg) * percentage)]
    print("距离界限:{}".format(del_line))
    # 开始删除最靠近边缘的10%样本
    #####
    del_ls = []
    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1 and distance(x[i], curr_Center) > del_line:
            # if train_k.cluster[i] == no :

            del_ls.append(i)
            # print("训练样本 %d 删除" % (i))
    ####
    print("本轮{}删除的多数类样本个数{}".format(no, len(del_ls)))
    print(del_ls)
    x = [x[i] for i in range(len(x)) if (i not in del_ls)]
    y = [y[i] for i in range(len(y)) if (i not in del_ls)]
    cluster = [cluster[i] for i in range(len(cluster)) if (i not in del_ls)]
    c = [c[i] for i in range(len(c)) if (i not in del_ls)]
    pos = [pos[i] for i in range(len(pos)) if (i not in del_ls)]
    u = [u[i] for i in range(len(u)) if (i not in del_ls)]
    train_k.x = np.array(x)
    train_k.y = np.array(y)
    train_k.cluster = cluster
    train_k.c = c
    train_k.pos = pos
    train_k.u = u


# Date:2022.05.05
# 删除这个簇中距离中心最远的样本且距离第二近的质心是正类
def kmean_delete_cluster_Neg_furthest_items_And_sec_cluster_Is_Pos(train_k, no, percentage=0.9):
    print("Delete cluster %d furthest Negtivate items" % no)
    x = train_k.x
    y = train_k.y
    c = train_k.c
    pos = train_k.pos
    u = train_k.u
    cluster = train_k.cluster
    # 记录样本到质心的距离
    d_neg = []
    # 记录样本到第二近的质心的距离（如果第二质心为正类，则真实记录，否则记为-1）
    sub_d_neg = []
    # 当前簇的质心
    curr_Center = train_k.x_c[no]
    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1:
            # if train_k.cluster[i] == no :
            # 放入第一近的质心
            d_neg.append(distance(x[i], curr_Center))
            # 找到第二近的质心
            # 初始第二近的质心为无穷大
            sub_item = float(inf)
            # 记录第二近的质心对应簇的序号
            J_no = -1
            for j in range(len(train_k.x_c)):
                # 之前的想法
                # if j !=no :
                if j != no and train_k.y_c[j] == 1:
                    curr_sub = distance(x[i], train_k.x_c[j])
                    if sub_item > curr_sub:
                        sub_item = curr_sub
                        J_no = j
                    # if j == 0:
                    #     print("finally:",distance(x[i], train_k.x_c[j]))
                # else:
                #     pass
            sub_d_neg.append(sub_item)
            # if train_k.y_c[J_no] == 1 :
            #     print("finally")
            # print(distance(x[i], curr_Center),sub_item,J_no)
    r = max(d_neg)
    # 上面的代码已经得到了最大半径和当前这个样本到最大半径的距离
    # 以及到最近的少数类质心的距离
    # 下面求的是我们需要删除的样本
    # # 记录样本到质心的距离
    # d_neg = []
    # # 记录样本到第二近的质心的距离（如果第二质心为正类）
    # sub_d_neg = []
    # 记录我们删除的目标函数的值和对应的样本位置
    del_item_value_index = dict()

    for i in range(x.shape[0]):
        if train_k.cluster[i] == no and train_k.y[i] == -1:
            # 放入第一近的质心
            first_item = distance(x[i], curr_Center)
            # 找到第二近的质心
            # 初始第二近的质心为无穷大
            sub_item = float(inf)
            # 记录第二近的质心对应簇的序号
            J_no = -1
            for j in range(len(train_k.x_c)):
                # 之前的想法
                # if j !=no :
                if j != no and train_k.y_c[j] == 1:
                    curr_sub = distance(x[i], train_k.x_c[j])
                    if sub_item > curr_sub:
                        sub_item = curr_sub
                        J_no = j

            last = first_item / r + sigmoid(sub_item - first_item)
            del_item_value_index[i] = last
            # print(first_item / r, 0.6*sigmoid(sub_item- first_item))
    del_item_value_index_2 = sorted(del_item_value_index.items(), key=lambda x: x[1], reverse=True)
    need_del = del_item_value_index_2[:int(len(del_item_value_index_2) * percentage)]
    print(need_del)
    del_ls = []
    for need_del_item in need_del:
        del_ls.append(need_del_item[0])
    ####
    print("本轮{}删除的多数类样本个数{}".format(no, len(del_ls)))
    print(del_ls)
    x = [x[i] for i in range(len(x)) if (i not in del_ls)]
    y = [y[i] for i in range(len(y)) if (i not in del_ls)]
    cluster = [cluster[i] for i in range(len(cluster)) if (i not in del_ls)]
    c = [c[i] for i in range(len(c)) if (i not in del_ls)]
    pos = [pos[i] for i in range(len(pos)) if (i not in del_ls)]
    u = [u[i] for i in range(len(u)) if (i not in del_ls)]
    train_k.x = np.array(x)
    train_k.y = np.array(y)
    train_k.cluster = cluster
    train_k.c = c
    train_k.pos = pos
    train_k.u = u


def kmean_predict1(train_k):
    acc = 0
    min_, sum_ = 0, 0
    y = train_k.y
    x = train_k.x
    train_k.count = [0] * train_k.k  # 各个簇中训练样本的数量
    train_k.count_acc = [0] * train_k.k  # 各个簇中预测正确的训练样本的数量
    train_k.sse = [0] * train_k.k  # 簇的密集程度
    train_k.k_c = [0] * train_k.k  # 每个簇的样本个数
    train_k.d = [0] * train_k.k

    for i in range(x.shape[0]):
        train_k.d[0] = min_ = distance(x[i], train_k.x_c[0])
        sum_ = 1.0 / train_k.d[0]
        train_k.c[i] = train_k.y_c[0]
        train_k.cluster[i] = 0
        # 以上为初始化
        for j in range(1, train_k.k):  # 从第二个质心开始
            train_k.d[j] = distance(train_k.x_c[j], train_k.x[i])
            if min_ > train_k.d[j]:
                min_ = train_k.d[j]
                train_k.c[i] = train_k.y_c[j]
                train_k.cluster[i] = j
            sum_ += 1.0 / train_k.d[j]
        train_k.count[train_k.cluster[i]] += 1
        train_k.k_c[train_k.cluster[i]] += 1
        train_k.sse[train_k.cluster[i]] += min_
        train_k.u[i] = (1.0 / train_k.d[train_k.cluster[i]]) / sum_

        if train_k.c[i] == train_k.y[i]:
            # 伪标签相关的代码，可以修改，这边没有进行修改
            acc += 1


# sigmod函数
def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def kmean_predict_testdata(train_k, test_k):
    acc = 0
    min_, sub_min_ = NAN, NAN
    x, y = test_k.x, test_k.y
    d1 = [NAN] * train_k.k
    for i in range(x.shape[0]):
        # //d放的是该样本到达每个质心的距离
        d1[0] = min_ = distance(train_k.x_c[0], x[i])
        test_k.c1[i] = train_k.y_c[0]
        test_k.cluster1[i] = 0

        for j in range(1, train_k.k):
            d1[j] = distance(train_k.x_c[j], x[i])

            if min_ > d1[j]:
                min_ = d1[j]
                test_k.c1[i] = train_k.y_c[j]
                test_k.cluster1[i] = j
        # // 计算对应簇的sse
        train_k.sse[test_k.cluster1[i]] += min_
        sub_min_ = 100000
        for j in range(train_k.k):
            if sub_min_ > d1[j] and fabs(min_ - d1[j]) >= 0.00001:
                sub_min_ = d1[j]
        test_k.u1[i] = sub_min_ - min_

    kmeans_del = average(test_k.u1)

    total = 0
    acc = 0
    for i in range(x.shape[0]):  # 对于test样本
        train_k.k_c[test_k.cluster1[i]] += 1  # //统计对应的簇的样本数量
        if test_k.u1[i] > kmeans_del:
            total += 1  # //满足条件（置信度大于平均值）的样本总数
            if test_k.c1[i] == y[i]: acc += 1

    train_k.total_sse = 0
    # 计算test中预测正确的个数
    # acc = sum(np.array(test_k.c1) == np.array(y))
    print("Kmean confident-test accuracy is %f,(%d/%d)" % (double(acc) / total, acc, total))

    print("Kmean test accuracy is %f,(%d/%d)" % (double(acc) / x.shape[0], acc, x.shape[0]))
    test_k.test_acc = double(acc) / x.shape[0]
    for j in range(train_k.k):  # /对于每个簇
        acc = 0
        for i in range(x.shape[0]):  # 对于每一个测试样本
            # // 如果测试样本的簇序号（通过模型预测来的）与该簇一致 // 如果测试样本的预测标签和其真实的标签一样，acc + 1
            if test_k.cluster1[i] == j and test_k.c1[i] == y[i]:
                acc += 1
        print("Cluster %d (Label %.0f) accuracy is %f,(%d/%d)" % (
            j, train_k.y_c[j], double(acc) / (train_k.k_c[j] - train_k.count[j]), acc,
            train_k.k_c[j] - train_k.count[j]), end="\t")

        train_k.total_sse += train_k.sse[j]
        if train_k.k_c[j] == 0:
            train_k.sse[j] = 0
        else:
            train_k.sse[j] /= train_k.k_c[j]
        print(" AVG SSE  %lf " % train_k.sse[j])
    train_k.total_sse /= x.shape[0]
    print("Total_AVG SSE %lf \n" % train_k.total_sse)
    return test_k.test_acc


def kmean_update_1(train_k, test_k):  # 对应main_jz中的kmean_update函数
    del_ = 0

    tmp_Center = copy.deepcopy(train_k.x_c)
    train_k.k_c = [0] * train_k.k
    sum_update = [0] * train_k.k
    for j in range(train_k.x.shape[0]):  # 对于训练样本
        # if train_k.use[j] == 0: continue
        train_k.k_c[train_k.cluster[j]] += 1
        sum_update[train_k.cluster[j]] += train_k.x[j]

    for j in range(test_k.x.shape[0]):  # 对于测试样本
        train_k.k_c[test_k.cluster1[j]] += 1
        sum_update[test_k.cluster1[j]] += test_k.x[j]

    for i in range(train_k.k):
        if train_k.k_c[i] == 0:
            train_k.x_c[i] = 0
        else:
            train_k.x_c[i] = sum_update[i] / train_k.k_c[i]
    for i in range(train_k.k):
        del_ += distance(tmp_Center[i], train_k.x_c[i])

    return del_


def kmean_update(train_k, test_k):
    del_ = 0

    tmp_Center = copy.deepcopy(train_k.x_c)
    for i in range(train_k.k):
        sum_this = 0  # 存放所有用于得到质心的点的sum
        train_k.k_c[i] = 0  # //每个簇的样本数量设置为0
        for j in range(train_k.x.shape[0]):  # 对于训练样本
            if train_k.y[j] == train_k.y_c[i]:  # //如果训练样本的值和样本的簇的类别一样的话
                # print("train\t%d\t%d"%( j, i))
                train_k.k_c[i] += 1
                sum_this += train_k.x[j]

        for j in range(test_k.x.shape[0]):  # //对于每个测试样本
            # // 如果测试样本是该簇的并且这个测试样本没有被选中作为伪标签样本
            if test_k.cluster1[j] == i and test_k.select_kmeans[j] == False:
                train_k.k_c[i] += 1
                sum_this += test_k.x[j]
        train_k.x_c[i] = sum_this / train_k.k_c[i]
    # // 得到新的质心和原始质心的距离差距
    for i in range(train_k.k):
        del_ += distance(tmp_Center[i], train_k.x_c[i])
    return del_


def iterative_update2(train_k, test_k, n):
    n1 = n + 1
    update_del = 100
    history_acc = train_k.history_acc
    history_SSE = train_k.history_SSE
    tmp_sum_SSE, tmp_sum_ACC = NAN, NAN
    kmean_predict0(train_k)
    weight = train_k.weight
    kmean_predict_testdata(train_k, test_k)
    tmp_sum_ACC = history_acc[n] = train_k.train_acc
    tmp_sum_SSE = history_SSE[n] = train_k.total_sse
    print("开始质心迭代！")

    while update_del > 0.001:  # //当质心不再发生变化
        # //保存旧的质心
        tmp_Center = copy.deepcopy(train_k.x_c)
        update_del = kmean_update(train_k, test_k)
        print(update_del)
        kmean_predict0(train_k)
        kmean_predict_testdata(train_k, test_k)
        history_acc[n1] = train_k.train_acc
        history_SSE[n1] = train_k.total_sse
        tmp_sum_ACC += history_acc[n1]
        tmp_sum_SSE += history_SSE[n1]
        t1 = weight * history_acc[n1 - 1] / tmp_sum_ACC - (1 - weight) * history_SSE[n1 - 1] / tmp_sum_SSE
        t2 = weight * history_acc[n1] / tmp_sum_ACC - (1 - weight) * history_SSE[n1] / tmp_sum_SSE
        print("update_del:%f,t1:%f\t t2:%f" % (update_del, t1, t2))

        if (t1 > t2):
            print("撤销本次质心迭代，恢复原质心!")
            train_k.x_c = copy.deepcopy(tmp_Center)
            break
        n1 += 1


def iterative_update2_1(train_k, test_k, n):
    n1 = n + 1
    update_del = inf
    weight = train_k.weight
    history_acc = train_k.history_acc
    history_SSE = train_k.history_SSE
    history_FVP = train_k.history_FVP
    history_DIS = train_k.history_DIS
    tmp_sum_SSE, tmp_sum_ACC, tmp_sum_FVP = NAN, NAN, NAN
    tmp_sum_DIS = NAN
    kmean_predict0(train_k)

    kmean_predict_testdata0(train_k, test_k)
    tmp_sum_ACC = history_acc[n] = train_k.train_acc
    tmp_sum_SSE = history_SSE[n] = train_k.total_sse
    tmp_sum_FVP = history_FVP[n] = train_k.train_f1
    tmp_sum_DIS = history_DIS[n] = train_k.Dis
    print("开始质心迭代！")

    while update_del > 0.001:  # //当质心不再发生变化
        # //保存旧的质心
        tmp_Center = copy.deepcopy(train_k.x_c)
        update_del = kmean_update_1(train_k, test_k)
        print(update_del)
        kmean_predict0(train_k)
        kmean_predict_testdata0(train_k, test_k)
        history_acc[n1] = train_k.train_acc
        history_SSE[n1] = train_k.total_sse
        history_FVP[n1] = train_k.train_f1
        history_DIS[n1] = train_k.Dis
        tmp_sum_ACC += history_acc[n1]
        tmp_sum_SSE += history_SSE[n1]
        tmp_sum_FVP += history_FVP[n1]
        tmp_sum_DIS += history_DIS[n1]
        # t1 = -history_SSE[n1 - 1] / tmp_sum_SSE + history_acc[n1 - 1] / tmp_sum_ACC + history_DIS[n1 - 1] / tmp_sum_DIS
        # t2 = -history_SSE[n1] / tmp_sum_SSE + history_acc[n1] / tmp_sum_ACC + history_DIS[n1] / tmp_sum_DIS
        t1 = -(1 - train_k.weight) * history_SSE[n1 - 1] / tmp_sum_SSE + train_k.weight * history_FVP[
            n1 - 1] / tmp_sum_FVP
        t2 = -(1 - train_k.weight) * history_SSE[n1] / tmp_sum_SSE + train_k.weight * history_FVP[n1] / tmp_sum_FVP
        print("update_del:%f,t1:%f\t t2:%f" % (update_del, t1, t2))
        if (t1 >= t2):
            print("撤销本次质心迭代，恢复原质心!")
            train_k.x_c = copy.deepcopy(tmp_Center)
            break
        n1 += 1


def kmean_train_new112(train_k, test_k):
    split_flg = True
    restore = False
    sum_SSE = sum_ACC = 0
    maxnum = train_k.maxnum
    need_split = [False] * maxnum
    history_acc = train_k.history_acc
    history_SSE = train_k.history_SSE
    n = 0
    weight = train_k.weight
    while n == 0 or (split_flg == True and train_k.k < maxnum):
        print("当前的n为：%d\n" % n)
        # // // // // // // // // // / 第一步：保存上次的聚类结果 // // // // // // // // // // // // // // // // // // // // // // // // // // //
        train_k.old_cluster = copy.deepcopy(train_k.cluster)  # 训练样本属于的簇序号
        train_k.old_c = copy.deepcopy(train_k.c)  # 训练样本的预测标签

        # test上的上次的预测结果
        test_k.old_c1 = copy.deepcopy(test_k.c1)
        # // // // // // // // // // / 第二步：进行测试 // // // // // // // // // // // // // // // // // // // // // // // // // // //
        kmean_predict0(train_k)
        # zly_sum_value = 0
        # for i in range(train_k.k):
        #     zly_sum_value += train_k.x_c[i][0]
        # print("zly_sum_value,",zly_sum_value)
        kmean_predict_testdata0(train_k, test_k)

        history_acc[n] = train_k.train_acc
        history_SSE[n] = train_k.total_sse
        sum_ACC += train_k.train_acc
        sum_SSE += train_k.total_sse
        # //////////////第三步：考察分裂效果，是否需要还原//////////////////////////////////////////
        if n != 0:
            tmp_k = copy.deepcopy(train_k.k)  # 保留分裂前的簇个数
            tmp_Center = copy.deepcopy(train_k.x_c)  # 保留分裂前的质心
            tmp_yc = copy.deepcopy(train_k.y_c)  # 质心标签
            tmp_cluster = copy.deepcopy(train_k.cluster)  # //将训练样本属于的簇序号暂时保存在tmp_cluster
            tmp_c = copy.deepcopy(train_k.c)  # ////训练样本的预测标签暂时保存在tmp_c

            restore = False

            i = 0
            while i < train_k.k:  # // 依次考察新分裂的簇，是否需要撤销
                if train_k.new_split[i] == 1:  # //是否新分裂的簇 0-不是；1-是；2-被新簇覆盖的旧的簇
                    # //如果新分裂的簇，他的准确率小于平均值（训练样本中各个簇的平均准确率）
                    if train_k.cluster_acc[i] < train_k.avg_cluster_acc:
                        # //要考虑两个条件（1.这个簇是不是新分裂出来的簇.2.这个簇的准确率是不是小于平均值）
                        if kmean_delete_centroid(train_k, i):
                            i -= 1
                            restore = True
                i += 1
            del i

            split_flg = False
            if 1 in train_k.new_split:
                split_flg = True
            # // 如果有新分裂的簇被撤销，则重新测试
            if split_flg == False and n != 0:
                break  # // 如果新分裂的簇都被撤销，停止循环

            if restore == True:
                print("重新测试")
                kmean_predict0(train_k)
                kmean_predict_testdata0(train_k, test_k)
                history_acc[n] = train_k.train_acc
                history_SSE[n] = train_k.total_sse
                sum_ACC += train_k.train_acc
                sum_SSE += train_k.total_sse
            print("history_acc(last and now):%lf  %lf" % (history_acc[n - 1], history_acc[n]))
            print("history_SSE(last and now):%lf   %lf" % (history_SSE[n - 1], history_SSE[n]))
            print("history_acc/sum_ACC(last and now)%lf   %lf" % (
                history_acc[n - 1] / sum_ACC, history_acc[n] / sum_ACC))
            print("history_SSE/sum_SSE(last and now)%lf   %lf" % (
                history_SSE[n - 1] / sum_SSE, history_SSE[n] / sum_SSE))

            t1 = weight * history_acc[n - 1] / sum_ACC - (1 - weight) * history_SSE[n - 1] / sum_SSE
            t2 = weight * history_acc[n] / sum_ACC - (1 - weight) * history_SSE[n] / sum_SSE

            if t1 >= t2:  # //撤销本次分裂，恢复原质心
                print("\n Restore former clusters!\n")
                train_k.k = copy.deepcopy(train_k.old_k)
                train_k.x_c = copy.deepcopy(Center)
                train_k.y_c = copy.deepcopy(train_k.old_yc)

                break

        if train_k.train_acc >= 1:
            print("train_acc 为1跳出")
            break
        dif = 0
        for i in range(test_k.x.shape[0]):  # // 统计两次预测的差异
            if test_k.old_c1[i] != test_k.c1[i]:
                dif += 1
        if n != 0 and dif < 1:
            print("两个模型预测test保持一样")
            break

        train_k.old_k = copy.deepcopy(train_k.k)

        Center = copy.deepcopy(train_k.x_c)
        train_k.old_yc = copy.deepcopy(train_k.y_c)
        train_k.old_sse = copy.deepcopy(train_k.sse)
        train_k.old_count = copy.deepcopy(train_k.count)
        train_k.old_count_acc = copy.deepcopy(train_k.count_acc)
        train_k.old_cluster_acc = copy.deepcopy(train_k.cluster_acc)

        # // // // // // // // 第三步：分裂 // // // // // // // // // // // //
        split_flg = False
        # //初始化所有簇的分裂状态
        train_k.parent = [-1] * maxnum
        # // 是否新分裂的簇
        # 0 - 不是；1 - 是；2 - 被新簇覆盖的旧的簇
        train_k.new_split = [0] * maxnum
        train_k.need_split = [False] * maxnum
        i = 0
        while i < train_k.old_k:
            # print("while")
            if train_k.cluster_acc[i] < 0.99 and train_k.k < train_k.maxnum:
                train_k.need_split[i] = True
                print("\t\tcluster_acc[%d]:%f\tn:%d\n" % (i, train_k.cluster_acc[i], n))
                if kmean_split1(train_k, test_k, i):
                    split_flg = True
            i += 1
        n += 1
    print("调出")
    kmean_predict0(train_k)
    kmean_predict_testdata0(train_k, test_k)
    history_acc[n] = train_k.train_acc
    history_SSE[n] = train_k.total_sse
    sum_ACC += train_k.train_acc
    sum_SSE += train_k.total_sse

    iterative_update2(train_k, test_k, n)


def kmean_train_new_1120(train_k, test_k):
    update_del = 100
    need_split = [False] * train_k.maxnum
    sum_SSE = sum_ACC = 0
    sum_FVP = 0
    sum_DIS = 0
    split_flg = True
    restore = False
    weight = train_k.weight
    history_acc = train_k.history_acc
    history_SSE = train_k.history_SSE
    history_FVP = train_k.history_FVP
    history_DIS = train_k.history_DIS
    n = 0
    while n == 0 or (split_flg == True and train_k.k < train_k.maxnum):
        print("当前的n为：%d\n" % n)
        #         保存上次的聚类结果
        train_k.old_cluster = copy.deepcopy(train_k.cluster)  # 训练样本属于的簇序号
        train_k.old_c = copy.deepcopy(train_k.c)  # 训练样本的预测标签

        # test上的上次的预测结果
        test_k.old_c1 = copy.deepcopy(test_k.c1)
        kmean_predict0(train_k)
        kmean_predict_testdata0(train_k, test_k)
        history_acc[n] = train_k.train_acc
        history_SSE[n] = train_k.total_sse
        history_FVP[n] = train_k.train_f1
        history_DIS[n] = train_k.Dis
        sum_ACC += history_acc[n]
        sum_SSE += history_SSE[n]
        sum_FVP += history_FVP[n]
        sum_DIS += history_DIS[n]
        # // // // // // // // // // / 保存上次的质心相关信息 // // // // // // // // // // // // // // // // // // // // // // // //
        train_k.old_k = copy.deepcopy(train_k.k)

        train_k.Center = copy.deepcopy(train_k.x_c)  # 保留分裂前的质心

        train_k.old_yc = copy.deepcopy(train_k.y_c)  # //保存分裂前的质心标签
        train_k.old_sse = copy.deepcopy(train_k.sse)
        train_k.old_cluster_acc = copy.deepcopy(train_k.cluster_acc)

        # // // // // // // // 第三步：分裂 // // // // // // // // // // // //
        if train_k.train_acc >= 1:
            print("train_acc 为1跳出")
            break
        dif = 0
        for i in range(test_k.x.shape[0]):  # // 统计两次预测的差异
            if test_k.old_c1[i] != test_k.c1[i]:
                dif += 1
        if n != 0 and dif < 1:
            print("两个模型预测test保持一样")
            break

        split_flg = False
        # // 初始化所有簇的分裂状态
        train_k.parent = [-1] * train_k.maxnum
        # // 是否新分裂的簇
        # 0 - 不是；1 - 是；2 - 被新簇覆盖的旧的簇
        train_k.new_split = [0] * train_k.maxnum
        train_k.need_split = [False] * train_k.maxnum

        # // 找出需要分裂的簇
        i = 0
        while i < train_k.old_k:
            # print("while")
            if train_k.cluster_acc[i] < 0.95 and train_k.k < train_k.maxnum:
                train_k.need_split[i] = True
                print("\t\tcluster_acc[%d]:%f\tn:%d\n" % (i, train_k.cluster_acc[i], n))
                if kmean_split1_1(train_k, test_k, i) == True:
                    split_flg = True
            i += 1
        del i
        if split_flg == True:
            kmean_predict0(train_k)
        # 考虑是否还原
        if n != 0:
            history_acc[n + 1] = train_k.train_acc
            history_SSE[n + 1] = train_k.total_sse
            history_FVP[n + 1] = train_k.train_f1
            history_DIS[n + 1] = train_k.Dis

            # d = (weight * history_FVP[n] / sum_FVP - (1 - weight) * history_SSE[n] / sum_SSE) - weight * (
            #         history_FVP[n + 1] / sum_FVP - (1 - weight) * history_SSE[n + 1] / sum_SSE)
            # d1 = - history_SSE[n] / sum_SSE + history_acc[n] / sum_ACC + history_DIS[n] / sum_DIS
            # d2 = - history_SSE[n + 1] / sum_SSE + history_acc[n + 1] / sum_ACC + history_DIS[n + 1] / sum_DIS
            d1 = -(1 - train_k.weight) * history_SSE[n] / sum_SSE + train_k.weight * history_acc[n] / sum_ACC
            d2 = - (1 - train_k.weight) * history_SSE[n + 1] / sum_SSE + train_k.weight * history_acc[n + 1] / sum_ACC
            d = d2 - d1
            if d2 < d1:
                print("\n Restore former clusters!")
                train_k.k = copy.deepcopy(train_k.old_k)
                train_k.x_c = copy.deepcopy(train_k.Center)

                train_k.y_c = copy.deepcopy(train_k.old_yc)
                kmean_predict0(train_k)
                kmean_predict_testdata0(train_k, test_k)
        n += 1
        print("本轮簇个数为:%d\n" % train_k.k)

    kmean_predict0(train_k)
    kmean_predict_testdata0(train_k, test_k)
    print("依次考察新分裂的簇")
    # todo
    i = 0
    while i < train_k.k:  # // 依次考察新分裂的簇，是否需要撤销
        if train_k.new_split[i] == 1:  # //是否新分裂的簇 0-不是；1-是；2-被新簇覆盖的旧的簇
            # //如果新分裂的簇，他的准确率小于平均值（训练样本中各个簇的平均准确率）
            if train_k.cluster_acc[i] < train_k.avg_cluster_acc and train_k.y_c[i] == -1:
                # //要考虑两个条件（1.这个簇是不是新分裂出来的簇.2.这个簇的准确率是不是小于平均值）
                if kmean_delete_centroid(train_k, i):
                    i -= 1
                    restore = True
        i += 1
    del i


#############################################     类分解部分代码      ###################################################

###############   指标计算相关    ###########

def getTargetClssPrecision(idx):
    return glo_param.tp[idx] / (glo_param.tp[idx] + glo_param.fp[idx])


def getTargetClsaaRecall(idx):
    return glo_param.tp[idx] / (glo_param.tp[idx] + glo_param.fn[idx])


def getTargetClassTNR(idx):
    return glo_param.tn[idx] / (glo_param.tn[idx] + glo_param.fp[idx])


def getTargetClassFScore(idx):
    x = 2 * getTargetClssPrecision(idx) * getTargetClsaaRecall(idx)
    y = getTargetClssPrecision(idx) + getTargetClsaaRecall(idx)
    return x / y


def getTargetClassGmean(idx):
    s1 = getTargetClsaaRecall(idx)
    s2 = getTargetClassTNR(idx)
    return math.sqrt(s1 * s2)


###############   end    ##################


def setTrainPosAndUse(model, prob):
    for i in range(0, prob.l):
        if (prob.r_y[i] == 1):
            prob.pos[i] = 1
        else:
            prob.pos[i] = 0
        prob.use[i] = 1


# 简化版：对于多分类不使用，多分类见C++版本
def initUseAndPosCluster(train_k, test_k):
    # 初始化k_param.use
    for i in range(train_k.k):
        train_k.use_k[i] = 1
    # 初始化k_param.pos
    for i in range(train_k.k):
        if train_k.y_c[i] == 1:
            train_k.pos[i] = 1
        else:
            train_k.pos[i] = 0


def kmean_delete_centroid_samples(train_k, test_k, prob, k_prob, no):
    f = False
    for i in range(train_k.k):
        if train_k.y_c[i] == train_k.y_c[no] and i != no:
            f = True
    if f == False:
        return False
    train_k.k -= 1
    print("Delete cluster %d;" % no)

    # 在删除该簇前，先删除该簇内的所有训练样本
    for i in range(prob.l):
        if train_k.cluster[i] == no:
            prob.use[i] = 0
            k_prob.use[i] = 0
            print("训练样本 %d 删除" % (i))

    # //对于每个簇，前移一位////
    train_k.k_c[no:] = train_k.k_c[no + 1:]
    train_k.y_c[no:] = train_k.y_c[no + 1:]
    train_k.x_c[no:] = train_k.x_c[no + 1:]

    train_k.size[no:] = train_k.size[no + 1:]
    print(train_k.pos[no:])
    print(train_k.pos[no + 1:])
    train_k.pos[no:] = train_k.pos[no + 1:]

    train_k.count[no:] = train_k.count[no + 1:]
    train_k.old_count[no:] = train_k.old_count[no + 1:]
    train_k.cluster_acc[no:] = train_k.cluster_acc[no + 1:]

    train_k.old_cluster_acc[no:] = train_k.old_cluster_acc[no + 1:]
    train_k.sse[no:] = train_k.sse[no + 1:]
    train_k.old_sse[no:] = train_k.old_sse[no + 1:]

    train_k.parent[no:] = train_k.parent[no + 1:]
    train_k.new_split[no:] = train_k.new_split[no + 1:]
    train_k.count_acc[no:] = train_k.count_acc[no + 1:]
    train_k.old_count_acc[no:] = train_k.old_count_acc[no + 1:]

    # 对于训练样本
    train_k.cluster = np.array(train_k.cluster)
    train_k.old_cluster = np.array(train_k.old_cluster)

    train_k.cluster[train_k.cluster == no] = train_k.maxnum - 1
    train_k.old_cluster[train_k.cluster == no] = train_k.maxnum - 1

    train_k.cluster[train_k.cluster > no] -= 1
    train_k.old_cluster[train_k.cluster > no] -= 1
    train_k.cluster = list(train_k.cluster)
    train_k.old_cluster = list(train_k.old_cluster)
    return True


def cluster_prepoocess(train_k, test_k, prob, k_prob, test_data, model):
    index1 = -1
    index2 = -1
    acc = 0
    no = 1
    min1 = 100000
    min2 = 100000
    id = -1
    flag = True

    while (flag):
        print("开始第%d次检查", no)
        no += 1
        # 移动聚类错误的少数类样本，删除聚类错误的多数类样本
        for i in range(train_k.k):
            if train_k.pos[i] == 0:  # 当前簇属于多数类
                for j in range(prob.l):
                    if prob.r_y[j] != train_k.y_c[i] and train_k.cluster[j] == i:  # 属于该簇，但是标签和此簇不同,那么此样本为少数类
                        min1 = 100000
                        for t in range(train_k.k):  # 寻找跟j最近的少数类簇并加入
                            if i != t and train_k.pos[t] == 1:
                                d = distance(train_k.x_c[t], prob.x[j])
                                if (d < min1):
                                    min1 = d
                                    id = t
                        # 添加到对应的少数类簇
                        temp = train_k.count[id]
                        train_k.c[j] = train_k.y_c[id]
                        train_k.cluster[j] = id
                        train_k.size[id] += 1
                        train_k.size[i] -= 1
                        train_k.k_c[i] -= 1
                        train_k.k_c[id] += 1
                        train_k.count[id] += 1
                        train_k.count[i] -= 1

                        print("簇%d中：训练样本:%d被移动，样本数:%d-->%d" % (i, j, temp, train_k.count[i]))
            if train_k.pos[i] == 1:  # 当前簇属于少数类
                for j in range(prob.l):
                    if prob.r_y[j] != train_k.y_c[i] and train_k.cluster[j] == i:  # 属于该簇，但是标签和此簇不同,那么此样本为多数类
                        t = train_k.count[i]
                        # 删除这些样本
                        train_k.size[i] -= 1
                        train_k.k_c[i] -= 1
                        train_k.count[i] -= 1
                        prob.use[j] = 0
                        train_k.cluster[j] = -1
                        print("簇%d中：训练样本:%d被删除，样本数:%d-->%d" % (i, j, t, train_k.count[i]))

        # 删除小尺寸的多数类簇
        i = 0
        while i < train_k.k:
            if train_k.pos[i] == 0 and train_k.size[i] < 3:
                kmean_delete_centroid_samples(train_k, test_k, prob, k_prob, i)
            i += 1
        del i
        # for i in range():
        #     if train_k.pos[i] == 0 and train_k.size[i] < 3:
        #         kmean_delete_centroid_samples(train_k, test_k, prob, k_prob, i)

        # 处理小尺寸的少数类簇（合并）
        for i in range(train_k.k):
            if train_k.pos[i] == 1 and train_k.size[i] < 3:
                min1 = min2 = 1000000
                for j in range(train_k.k):
                    if i != j and min1 > distance(train_k.x_c[i], train_k.x_c[j]):  # 寻找跟该簇距离最近的簇
                        min1 = distance(train_k.x_c[i], train_k.x_c[j])
                        index1 = j
                    if i != j and train_k.y_c[i] == train_k.y_c[j]:  # 寻找跟该簇同类别且距离最近的簇
                        if min2 > distance(train_k.x_c[i], train_k.x_c[j]):
                            min2 = distance(train_k.x_c[i], train_k.x_c[j])
                            index2 = j
                if index1 == index2 and index1 != -1:  # 如果距离最近的簇是同类别的，则直接合并
                    train_k.k -= 1
                    t = train_k.count[index1]
                    # 对于每个簇，前移一位
                    train_k.k_c[j:] = train_k.k_c[j + 1:]
                    train_k.y_c[j:] = train_k.y_c[j + 1:]
                    train_k.x_c[j:] = train_k.x_c[j + 1:]

                    train_k.size[j:] = train_k.size[j + 1:]
                    train_k.pos[j:] = train_k.pos[j + 1:]

                    train_k.count[j:] = train_k.count[j + 1:]
                    train_k.old_count[j:] = train_k.old_count[j + 1:]
                    train_k.cluster_acc[j:] = train_k.cluster_acc[j + 1:]

                    train_k.old_cluster_acc[j:] = train_k.old_cluster_acc[j + 1:]
                    train_k.sse[j:] = train_k.sse[j + 1:]
                    train_k.old_sse[j:] = train_k.old_sse[j + 1:]

                    train_k.new_split[j:] = train_k.new_split[j + 1:]
                    train_k.count_acc[j:] = train_k.count_acc[j + 1:]
                    train_k.old_count_acc[j:] = train_k.old_count_acc[j + 1:]

                    # 删除完当前的簇之后，判断其合并到的簇的索引是否变化
                    if index1 > i:
                        index1 = index1 - 1
                        index2 = index2 - 1

                    # 对于训练样本
                    for j in range(prob.l):
                        if train_k.cluster[j] == i:
                            train_k.cluster[j] = index1
                            train_k.size[index1] += 1
                            train_k.k_c[index1] += 1
                            train_k.count[index1] += 1
                            train_k.c[j] = train_k.y_c[index1]
                        elif train_k.cluster[j] > i:
                            train_k.cluster[j] = train_k.cluster[j] - 1
                            train_k.old_cluster[j] = train_k.old_cluster[j] - 1
                            train_k.c[j] = train_k.y_c[train_k.cluster[j]]
                    # 对于测试样本
                    for j in range(test_data.l):
                        if test_k.cluster1[j] == i:
                            test_k.cluster1[j] = index1
                            train_k.size[index1] += 1
                            test_k.c1[j] = train_k.y_c[index1]
                        if test_k.cluster1[j] > i:
                            test_k.cluster1[j] = test_k.cluster1[j] - 1
                            test_k.c1[j] = test_k.c1[test_k.cluster1[j]]

                    print("(合并)簇%d--->簇%d，样本个数%d-->%d" % (i, index1, t, test_k.count1[index1]))
        kmean_predict_testdata0(train_k, test_k)  # 无标签的样本进行预测 将测试样本分配到合适的簇中

        # 检查当前所有簇是否都合格
        flag = False
        for i in range(train_k.k):
            print("簇%d的总样本数%d\n" % (i, train_k.size[i]))
            if train_k.size[i] < 3 and train_k.pos[i] == 0:
                flag = True
                break

        print("目前共剩下簇 %d 个" % (train_k.k))
        glo_param.useClusterNum = train_k.k

        print("簇处理完成后，各个簇的情况：")
        for i in range(train_k.k):
            if train_k.use_k[i] == 0:
                continue
            nn = 0
            acc = 0

            # 计算测试集的ACC
            for j in range(test_data.l):
                if test_k.cluster1[j] == i:
                    nn += 1
                    if test_k.c1[j] == test_data.r_y[j]:
                        acc += 1

            # 计算训练集的ACC
            train_k.count_acc[i] = 0
            for j in range(prob.l):
                if prob.use[j] == 0 or k_prob.use[j] == 0:
                    continue
                if train_k.cluster[j] == i and prob.r_y[j] == train_k.c[j]:
                    train_k.count_acc[i] += 1
            if train_k.count[i] == 0:
                train_k.count_acc[i] = 0
            else:
                train_k.cluster_acc[i] = train_k.count_acc[i] / train_k.count[i]
            print(
                "训练集：Cluster %d (Label %.0f) accuracy is %f,(%d/%d/%d)" % (i, train_k.y_c[i], train_k.cluster_acc[i],
                                                                              train_k.count_acc[i], train_k.count[i],
                                                                              train_k.size[i]))


def getAvgSubclass(train_k, oriSample):
    trueNum = 0
    for i in range(train_k.k):
        for j in range(oriSample.l):
            if glo_param.sub_cluster[j] == i and oriSample.r_y[j] == glo_param.sub_c[j]:
                trueNum += 1
    tSub = trueNum / glo_param.useClusterNum
    glo_param.avgSubclass = math.floor(tSub + 0.5) if tSub > 0.0 else math.ceil(tSub - 0.5)
    print("子类的平均大小是：", glo_param.avgSubclass)


def getEachSubclassAcc(addSamples, train_k):
    t_acc = 0
    for i in range(addSamples.l):
        if glo_param.sub_c[i] == addSamples.r_y[i]:
            t_acc += 1
    print("半监督聚类中训练样本ACC:", t_acc / addSamples.l)

    for i in range(train_k.k):
        train_k.count_acc[i] = 0
        if train_k.use_k[i] == 0:
            continue
        for j in range(addSamples.l):
            if addSamples.use[j] == 0:
                continue
            if glo_param.sub_cluster[j] == i and addSamples.r_y[j] == glo_param.sub_c[j]:
                train_k.count_acc[i] += 1
        if glo_param.sub_count[i] == 0:
            train_k.cluster_acc[i] = 0
        else:
            train_k.cluster_acc[i] = train_k.count_acc[i] / glo_param.sub_count[i]
        print("Cluster %d (Label %.0f) accuracy is %f,(%d/%d)" % (
            i, train_k.y_c[i], train_k.cluster_acc[i], train_k.count_acc[i], glo_param.sub_count[i]))


def reSamplingSmote(train_k, test_k, prob, test_data, model):
    # oriSample = svm_problem([], []) # 保留簇中预测正确的训练样本
    # pseudoSamples = svm_problem([], []) # 添加的伪标签样本
    # smoteSamples = svm_problem([], []) # 添加的SMOTE生成样本
    addSamples = svm_problem([], [])  # 最终添加的样本数

    # 仅保留各个簇中预测正确的训练样本
    for i in range(train_k.k):
        if train_k.use_k[i] == 0:
            continue
        if train_k.k_c[i] == 0 or train_k.count[i] == 0:
            continue
        glo_param.sub_count.append(0)
        train_k.k_c[i] = 0
        train_k.size[i] = 0  # 三者都是记录子簇所含样本数量
        for j in range(prob.l):
            # sub_prob = svm_problem([], [])
            if prob.use[j] == 0:
                continue
            if train_k.cluster[j] == i and prob.r_y[j] == train_k.c[j]:  # 属于该簇且预测正确的训练样本
                addSamples.r_y.append(prob.r_y[j])  # append进去的变量被修改后，添加过的数组中的样本是否会变化
                addSamples.y.append(prob.y[j])
                addSamples.pos.append(prob.pos[j])
                addSamples.use.append(prob.use[j])
                addSamples.x.append(prob.x[j])
                glo_param.sub_cluster.append(train_k.cluster[j])
                glo_param.sub_c.append(train_k.c[j])
                glo_param.sub_count[i] += 1
                train_k.k_c[i] += 1
                train_k.size[i] += 1
                addSamples.l += 1
    # print(oriSample)
    print("保留预测正确的训练样本个数为：", addSamples.l)

    getAvgSubclass(train_k, addSamples)  # 获得子类（簇）的平均大小

    # 下面开始进行重采样 - -根据avgSubclass

    # step1: 过采样：选择置信度高的测试样本到该子类中
    for i in range(train_k.k):
        if glo_param.sub_count[i] <= 0:
            continue
        if train_k.use_k[i] == 0:
            continue
        dis = []

        # 方式1：根据测试样本到当前质心的距离筛选高置信度
        for j in range(test_data.l):
            if test_k.cluster1[j] == i:
                temp = sortSample()
                temp.id = j
                temp.d = distance(train_k.x_c[i], test_data.x[j])
                dis.append(temp)

        dis.sort(key=lambda x: x.d)
        dif_sub_avg = glo_param.sub_count[i] - glo_param.avgSubclass  # 判断子类是进行过采样还是欠采样
        addTestNum = int(len(dis) * 0.1)  # 伪标签选取的比例

        if addTestNum > 0 and dif_sub_avg < 0 and train_k.y_c[i] == 1:
            # step1.首先添加距离质心最近的10%
            for j in range(addTestNum):
                tId = dis[j].id  # 按距离从小到大排列后的样本索引
                addSamples.r_y.append(1)
                addSamples.y.append(1)
                addSamples.pos.append(1)
                addSamples.use.append(1)
                addSamples.x.append(test_data.x[tId])
                glo_param.sub_cluster.append(test_k.cluster1[tId])
                glo_param.sub_c.append(test_k.c1[tId])
                glo_param.sub_count[i] += 1
                train_k.k_c[i] += 1
                train_k.size[i] += 1
                addSamples.l += 1
        # print("添加伪标签样本个数为：", addSamples.l)

        dif_sub_avg = glo_param.sub_count[i] - glo_param.avgSubclass  # 判断添加测试样本后，是否达到平衡

        # 开始进行SMOTE
        if dif_sub_avg < 0 and train_k.y_c[i] == 1:
            # step1:确定采样倍率，最近邻K，并筛选哪些样本需要进行SMOTE
            KN = glo_param.tK  # 默认为3
            N = abs(dif_sub_avg)  # 随机选的样本个数，即采样倍率
            idInCluster = []  # 当前簇中样本的id
            selectId = []  # 被选中进行采样的样本id
            # 获得当前簇中的样本Id
            for j in range(addSamples.l):
                if glo_param.sub_cluster[j] == i:
                    idInCluster.append(j)
            # 随机选取进行采样的样本Id
            for j in range(N):
                sId = random.randint(0, len(idInCluster) - 1)
                selectId.append(idInCluster[sId])  # 注意添加的序号是：idInCluster[sId]，sId只是在idInCluster中的索引

            # step2: 对于每一个需要被SMOTE的样本，计算其K近邻
            for j in range(N):
                curId = selectId[j]  # curId表示基准样本的id,即addSamples中的id
                kNearst = []
                # 个别簇样本数少于3，导致取最近邻时越界
                if len(idInCluster) > 2:
                    for k in range(len(idInCluster)):
                        if idInCluster[k] != curId:
                            temp = sortSample()
                            temp.id = idInCluster[k]
                            temp.d = distance(addSamples.x[temp.id], addSamples.x[curId])
                            kNearst.append(temp)
                else:
                    for k in range(len(idInCluster)):
                        temp = sortSample()
                        temp.id = idInCluster[k]
                        temp.d = distance(addSamples.x[temp.id], addSamples.x[curId])
                        kNearst.append(temp)
                    # 簇内样本数过少，将质心也添加进去
                    temp = sortSample()
                    temp.id = -1
                    temp.d = distance(train_k.x_c[i], addSamples.x[curId])
                    kNearst.append(temp)

                KN = len(kNearst) if KN > len(kNearst) else KN
                kNearst.sort(key=lambda x: x.d)

                # step3: 根据公式生成新的样本并添加到数据集中
                tId = random.randint(0, KN - 1)
                sId = kNearst[tId].id  # 被选中的近邻样本的id
                if sId != -1:
                    newSample = svm_problem([], [])
                    newSample.x.append(diminish(addSamples.x[curId], addSamples.x[sId]))
                    factor = random.random()
                    newSample.x[0] = multiply(factor, newSample.x[0])
                    newSample.x[0] = add1(newSample.x[0], addSamples.x[
                        curId])  # np.array(newSample.x[0]) + np.array(addSamples.x[curId])

                    # 这里面curId是相对于sub_prob的索引来说的，所以不能使用cluster[curId] c[curId]
                    addSamples.r_y.append(addSamples.r_y[curId])
                    addSamples.y.append(addSamples.y[curId])
                    addSamples.pos.append(addSamples.pos[curId])
                    addSamples.use.append(addSamples.use[curId])
                    addSamples.x.append(newSample.x[0])
                    glo_param.sub_cluster.append(i)
                    glo_param.sub_c.append(1)
                    glo_param.sub_count[i] += 1
                    train_k.k_c[i] += 1
                    train_k.size[i] += 1
                    addSamples.l += 1
                else:
                    newSample = svm_problem([], [])
                    newSample.x.append(diminish(addSamples.x[curId], train_k.x_c[i]))
                    factor = random.random()
                    newSample.x[0] = multiply(factor, newSample.x[0])
                    newSample.x[0] = add1(newSample.x[0], addSamples.x[
                        curId])  # np.array(newSample.x[0]) + np.array(addSamples.x[curId])

                    # 这里面curId是相对于sub_prob的索引来说的，所以不能使用cluster[curId] c[curId]
                    addSamples.r_y.append(train_k.y_c[i])
                    addSamples.y.append(train_k.y_c[i])
                    addSamples.pos.append(1)
                    addSamples.use.append(1)
                    addSamples.x.append(newSample.x[0])
                    glo_param.sub_cluster.append(i)
                    glo_param.sub_c.append(1)
                    glo_param.sub_count[i] += 1
                    train_k.k_c[i] += 1
                    train_k.size[i] += 1
                    addSamples.l += 1
                kNearst.clear()
            selectId.clear()
            idInCluster.clear()

    print("重采样后各子类的准确率：")
    getEachSubclassAcc(addSamples, train_k)
    # 将样本数为0的簇删除
    for i in range(train_k.k):
        if glo_param.sub_count[i] <= 0:
            train_k.use_k[i] = 0
    return addSamples


def getOriLabel(label):
    oriLabel = 1
    flag = False
    for key in glo_param.hashMap.keys():
        psoLabels = glo_param.hashMap.get(key)
        for yy in psoLabels:
            if yy == label:
                oriLabel = key
                flag = True
                break
        if flag:
            break
    return oriLabel


def getSvmTrainPerformance(train_k, test_k, prob, sub_prob, model):
    max_p = -100
    max_index = -1
    acc = 0
    svm_del = 0
    temp_confidence = []

    # 是预测sub_prob还是prob？
    for i in range(prob.l):
        temp_confidence = model.predict_proba(prob.x[i].reshape(1, -1))
        temp_confidence = temp_confidence[0]
        max_p = temp_confidence[0]
        max_index = 0
        for t in range(len(model.classes_)):
            if max_p < temp_confidence[t]:
                max_p = temp_confidence[t]
                max_index = t
        prob.y[i] = getOriLabel(model.classes_[max_index])
    # print(accuracy_score(prob.r_y, prob.y))

    # 计算SVM相关指标
    # 初始化为0
    for i in range(train_k.maxnum):
        glo_param.tp[i] = 0
        glo_param.fp[i] = 0
        glo_param.tn[i] = 0
        glo_param.fn[i] = 0
    for i in range(len(glo_param.oriLabelIdx)):
        for j in range(prob.l):
            if prob.r_y[j] == glo_param.oriLabelIdx[i] and prob.y[j] == glo_param.oriLabelIdx[i]:
                glo_param.tp[i] += 1
            if prob.y[j] == glo_param.oriLabelIdx[i] and prob.r_y[j] != glo_param.oriLabelIdx[i]:
                glo_param.fp[i] += 1
            if prob.r_y[j] == glo_param.oriLabelIdx[i] and prob.y[j] != glo_param.oriLabelIdx[i]:
                glo_param.fn[i] += 1
            if prob.r_y[j] != glo_param.oriLabelIdx[i] and prob.y[j] != glo_param.oriLabelIdx[i]:
                glo_param.tn[i] += 1
    id = 0 if glo_param.oriLabelIdx[0] == 1 else 1
    glo_param.svm_train_f1 = getTargetClassFScore(id)  # 这里直接计算类别1的f1-value
    glo_param.svm_train_gm = getTargetClassGmean(id)  # 这里直接计算类别1的gm
    print("SVM在训练集上的f1:%f  gm:%f" % (glo_param.svm_train_f1, glo_param.svm_train_gm))

    # 计算kmeans的相关指标
    # 初始化为0
    for i in range(train_k.maxnum):
        glo_param.tp[i] = 0
        glo_param.fp[i] = 0
        glo_param.tn[i] = 0
        glo_param.fn[i] = 0
    for i in range(len(glo_param.oriLabelIdx)):
        for j in range(prob.l):
            if prob.r_y[j] == glo_param.oriLabelIdx[i] and train_k.c[j] == glo_param.oriLabelIdx[i]:
                glo_param.tp[i] += 1
            if train_k.c[j] == glo_param.oriLabelIdx[i] and prob.r_y[j] != glo_param.oriLabelIdx[i]:
                glo_param.fp[i] += 1
            if prob.r_y[j] == glo_param.oriLabelIdx[i] and train_k.c[j] != glo_param.oriLabelIdx[i]:
                glo_param.fn[i] += 1
            if prob.r_y[j] != glo_param.oriLabelIdx[i] and train_k.c[j] != glo_param.oriLabelIdx[i]:
                glo_param.tn[i] += 1
    id = 0 if glo_param.oriLabelIdx[0] == 1 else 1
    glo_param.kmean_train_f1 = getTargetClassFScore(id)  # 这里直接计算类别1的f1-value
    glo_param.kmean_train_gm = getTargetClassGmean(id)  # 这里直接计算类别1的gm
    print("kmeans在训练集上的f1:%f gm:%f" % (glo_param.kmean_train_f1, glo_param.kmean_train_gm))


def mergeSubPro(test_data, svm_confidence, model):
    """
    svm_ori_confidence[i][posId]: 父标签为正类的概率
	svm_ori_confidence[i][negId]：父标签为负类的概率
	保证svm_ori_confidence与kmeans_confidence正负类样本对应标签的概率分布一致
    """

    posId = -1;
    negId = -1  # 正负类样本的位置
    svm_ori_confidence = []
    nr_class = len(model.classes_)
    for i in range(len(glo_param.oriLabelIdx)):
        if glo_param.oriLabelIdx[i] == 1:
            posId = i
        if glo_param.oriLabelIdx[i] == -1:
            negId = i

    for i in range(test_data.l):
        svm_ori_confidence.append([0, 0])
        for j in range(nr_class):
            if getOriLabel(model.classes_[j]) == 1:
                svm_ori_confidence[i][posId] += svm_confidence[i][j]
            if getOriLabel(model.classes_[j]) == -1:
                svm_ori_confidence[i][negId] += svm_confidence[i][j]
    return svm_ori_confidence


def final_predict3(train_k, test_k, prob, sub_prob, test_data, model):
    final_result = [NAN] * test_data.l
    svm_confidence = []
    max_p = -100
    max_index = -1
    acc = 0
    svm_del = 0
    nr_class = len(model.classes_)
    for i in range(test_data.l):
        # model.predict(test_data.x[i].reshape(1, -1))
        temp_confidence = model.predict_proba(test_data.x[i].reshape(1, -1))
        temp_confidence = temp_confidence[0]
        # print(temp_confidence)
        svm_confidence.append(temp_confidence)
        max_p = temp_confidence[0]
        max_index = 0
        for t in range(1, nr_class):
            if max_p < temp_confidence[t]:
                max_p = temp_confidence[t]
                max_index = t
        test_data.p[i] = max_p
        svm_del += max_p
        test_data.y[i] = getOriLabel(model.classes_[max_index])  # TODO

    # print(accuracy_score(test_data.r_y, test_data.y))

    # 方式2：以SVM，kmeans二者的f1或G-mean的结果为权重
    getSvmTrainPerformance(train_k, test_k, prob, sub_prob, model)  # 获取仅使用SVM的性能参数
    new_weight = glo_param.kmean_train_f1 / (glo_param.kmean_train_f1 + glo_param.svm_train_f1)

    svm_ori_confidence = mergeSubPro(test_data, svm_confidence, model)
    for i in range(test_data.l):
        id = 0 if glo_param.oriLabelIdx[0] == 1 else 1
        test_data.p[i] = svm_ori_confidence[i][id]
        test_data.r_y_p[i] = svm_ori_confidence[i][id]

    """
    重点：
		svm_ori_confidence：记录的是样本在父标签(-1,1)上的预测概率
		svm_ori_confidence[i][posId]: 父标签为正类的概率
		svm_ori_confidence[i][negId]：父标签为负类的概率
		保证svm_ori_confidence与kmeans_confidence正负类样本对应标签的概率分布一致
    """

    kmeans_confidence = test_k.kmeans_confidence
    for i in range(test_data.l):
        if test_k.c1[i] == test_data.y[i]:
            final_result[i] = test_data.y[i]
        else:
            max_p = 0
            max_index = -1
            nrClass = 2
            for t in range(nrClass):
                print("kmeans:%f  svm:%f" % (svm_ori_confidence[i][t], kmeans_confidence[i][t]))
                if max_p < (1 - new_weight) * svm_ori_confidence[i][t] + new_weight * kmeans_confidence[i][t]:
                    max_p = (1 - new_weight) * svm_ori_confidence[i][t] + new_weight * kmeans_confidence[i][t]
                    max_index = t

            final_result[i] = getOriLabel(model.classes_[max_index])
            if test_data.y[i] != final_result[i]:
                id = 0 if glo_param.oriLabelIdx[0] == 1 else 1
                print("测试样本%d预测标签有变化, %.0f-->%.0f" % (i, test_data.y[i], final_result[i]))
                test_data.r_y_p[i] = (1 - new_weight) * svm_ori_confidence[i][id] + new_weight * kmeans_confidence[i][
                    id]
            test_data.y[i] = final_result[i]

        if final_result[i] == test_data.r_y[i]:
            acc += 1
        else:
            print("第%d个样本，标签是%.0f" % (i, test_data.r_y[i]))

    print("Final Acc:%f (%d/%d)" % (acc / test_data.l, acc, test_data.l))
    final_acc = acc / test_data.l
    return final_acc


def changeToOriLabel(test_data, model):
    flag = False
    keys = glo_param.hashMap.keys()
    for key in keys:
        print(key, end=":")
        for label in glo_param.hashMap[key]:
            print(label, end=" ")
        print()
    for i in range(test_data.l):
        flag = False
        for key in keys:
            for label in glo_param.hashMap[key]:
                if label == test_data.y[i]:
                    test_data.y[i] = key
                    flag = True
            if flag:
                break


# 交叉验证产生数据集
def Stratified_fold_K(x, y, n_spli=5, random_sta=42):
    skf = StratifiedKFold(n_splits=n_spli, shuffle=True, random_state=random_sta)
    i = 0
    try:
        os.mkdir("data_mid")
    except OSError as error:
        pass
    for train_index, test_index in skf.split(x, y):
        i = i + 1
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        savelibsvm(X_train, y_train, "data_mid/{}train.txt".format(i))
        savelibsvm(X_test, y_test, "data_mid/{}test.txt".format(i))


# 交叉验证产生数据集
# 如果use_file为空表示不使用本地文件创建的方式，不是空表示返回一个字典包含4个list（train_x,train_y和test_x和test_y）
def Stratified_fold_K_version_2(x, y, n_spli=5, use_file=None, random_sta=42):
    skf = StratifiedKFold(n_splits=n_spli, shuffle=True, random_state=random_sta)
    i = 0
    if use_file != None:
        try:
            os.mkdir("{}".format(use_file))
        except OSError as error:
            pass
        for train_index, test_index in skf.split(x, y):
            i = i + 1
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            savelibsvm(X_train, y_train, "{}/{}train.txt".format(use_file, i))
            savelibsvm(X_test, y_test, "{}/{}test.txt".format(use_file, i))
    else:
        res = {}
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for train_index, test_index in skf.split(x, y):
            i = i + 1
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_x.append(copy.deepcopy(X_train))
            train_y.append(copy.deepcopy(y_train))
            test_x.append(copy.deepcopy(X_test))
            test_y.append(copy.deepcopy(y_test))
        res["train_x"] = train_x
        res["train_y"] = train_y
        res["test_x"] = test_x
        res["test_y"] = test_y
        return res


# def savelibsvm(x, y, filename):
#     f = open(filename, "w", encoding='UTF-8')
#     m, n = x.shape
#     for index in range(m):
#         out = str(y[index])
#         for i in range(n):
#             out = out + "\t{}:{}".format(i + 1, x[index, i])
#         out = out + "\n"
#         f.writelines(out)
#     f.close()
def savelibsvm(x, y, filename):
    dump_svmlight_file(x, y, filename)


# 计算每个簇的Gmeans,删除低于阈值的簇
def ProcessEveryClusterGmeans(train_k):
    # // 找出需要删除的簇
    i = 0
    while i < train_k.k:  # // 依次考察新分裂的簇，是否需要撤销

        # //如果新分裂的簇，他的准确率小于平均值（训练样本中各个簇的平均准确率）
        if train_k.cluster_acc[i] < 1.0 / len(set(train_k.y)):
            # //要考虑两个条件（1.这个簇是不是新分裂出来的簇.2.这个簇的准确率是不是小于平均值）
            if kmean_delete_centroid(train_k, i):
                i -= 1
                restore = True
        i += 1
    del i


# 这边对应算法2的第二部分
def UnderSamplingMajority(train_k, test_k, min_size=10, percentage=0.9):
    # 删除小尺寸的多数类簇
    kmean_predict0(train_k)
    kmean_predict_testdata0_no_show(train_k, test_k)
    print("删除小尺寸多数类簇(小于{})".format(min_size))
    i = 0
    while i < train_k.k:
        # TODO 需要设置参数的地方
        if train_k.y_c[i] == -1 and train_k.size[i] < min_size:
            if kmean_delete_cluster_And_items(train_k, i) == True:
                i -= 1
                kmean_predict0(train_k)
                kmean_predict_testdata0_no_show(train_k, test_k)
        i += 1
    kmean_predict0(train_k)
    kmean_predict_testdata0_no_show(train_k, test_k)
    del i
    print("对于每一个不纯的少数类簇，删去其中所有的多数类")

    # 对于每一个不纯的少数类簇，删去其中所有的多数类
    i = 0
    while i < train_k.k:
        if train_k.y_c[i] == 1:
            kmean_delete_minority_items(train_k, i)
        i += 1
    del i
    kmean_predict0(train_k)
    kmean_predict_testdata0_no_show(train_k, test_k)
    print("修改多数类簇中，位于边缘的点")
    i = 0
    while i < train_k.k:
        # TODO 需要设置参数的地方
        if train_k.y_c[i] == -1:
            kmean_delete_cluster_Neg_furthest_items_And_sec_cluster_Is_Pos(train_k, i, percentage=percentage)
            # kmean_delete_cluster_Neg_furthest_items(train_k, i, percentage=percentage)
            # kmean_change_cluster_Neg_furthest_items(train_k, i, percentage=percentage)
        i += 1
    kmean_predict0(train_k)
    kmean_predict_testdata0_no_show(train_k, test_k)
    del i


# x:列向量,dim:将列向量从1列映射到dim列,sigma自己设定的
def rbfmap(x, dim, sigma):
    X = np.zeros((x.shape[0], dim))
    for i in range(dim):
        s = 1
        for j in range(i + 1):
            if j == 0:
                j = 1
            s = s * j
        t = sigma ** (2 * i)

        for n in range(len(x)):
            xx1 = x[n, 0] * x[n, 0]
            e = math.exp(- xx1 / (2 * sigma ** 2))
            xx2 = x[n, 0] ** i
            s = math.sqrt(1 / (s * t)) * e * xx2
            X[n, i] = s
    return X


def OverSamplingMinority(train_k, test_k, percentage=0.5):
    # 记录未处理前的样本个数
    over_before_num = train_k.x.shape[0]
    over_before_x = copy.deepcopy(train_k.x)
    over_before_y = copy.deepcopy(train_k.y)
    # 将测试样本中距离少数类质心最近的样本作为训练集添加
    #
    for i in range(train_k.k):
        if train_k.y_c[i] == 1:
            choose_unlabeled_data(train_k, test_k, i, percentage=percentage)
    # 如果少数类和多数类差距还是过大，用smote添加样本(选择原始样本)
    ori_len = train_k.x.shape[0]
    min_aim_num = int(1.0 * Counter(over_before_y)[-1]) - (ori_len - over_before_num)
    # BorderlineSMOTE()
    # 这边smote修改成BorderlineSMOTE
    smo = SMOTE(sampling_strategy={-1: Counter(over_before_y)[-1], 1: min_aim_num}, k_neighbors=5, random_state=42)
    # X_smo, y_smo = smo.fit_resample(over_before_x, over_before_y)
    X_smo, y_smo = smo.fit_resample(train_k.x[:ori_len], train_k.y[:ori_len])
    smote_add_x = X_smo[over_before_num:]
    smote_add_y = y_smo[over_before_num:]
    train_k.x = _extend_(train_k.x, smote_add_x)
    train_k.y = _extend_(train_k.y, smote_add_y)
    print(Counter(train_k.y))


# 选择伪标签作为少数类添加到train中
def choose_unlabeled_data(train_k, test_k, no, percentage=0.5):
    x1 = test_k.x

    ls = []
    for i in range(x1.shape[0]):
        if test_k.cluster1[i] == no:
            ls.append(distance(x1[i], train_k.x_c[no]))
    # todo 添加置信度最高的70%样本
    ls.sort(reverse=True)
    confident_line = -1
    if len(ls) != 0:
        if int(len(ls) * percentage) == len(ls):
            confident_line = ls[int(len(ls) * percentage) - 1]
        else:
            confident_line = ls[int(len(ls) * percentage)]
    add_num = 0
    true_add_num = 0

    print(confident_line)
    if confident_line != -1:
        for i in range(x1.shape[0]):
            if test_k.cluster1[i] == no and test_k.c1[i] == 1 and test_k.u1[i] > confident_line:
                # print("true:{}\tpred:{}".format(test_k.y[i],test_k.c1[i]))
                if test_k.y[i] == test_k.c1[i]:
                    true_add_num += 1
                add_num += 1
                train_k.x = _append_(train_k.x, x1[i])
                train_k.y = _append_(train_k.y, 1)
                train_k.k_c[no] += 1
    print("簇{}添加了{}个伪标签样本({}/{})".format(no, add_num, true_add_num, add_num))


def kernelTrans(X, A, kTup):
    """
    Function：   核转换函数

    Input：      X：数据集
                A：某一行数据
                kTup：核函数信息

    Output： K：计算出的核向量
    """
    # 获取数据集行列数
    m, n = shape(X)
    # 初始化列向量
    K = mat(zeros((m, 1)))
    # 根据键值选择相应核函数
    # lin表示的是线性核函数
    if kTup[0] == 'lin':
        K = X * A.T
    # rbf表示径向基核函数
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 对矩阵元素展开计算，而不像在MATLAB中一样计算矩阵的逆
        K = exp(K / (-1 * kTup[1] ** 2))
    # 如果无法识别，就报错
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    # 返回计算出的核向量
    return K


def MWMOTE(x, y):
    Smin = pd.DataFrame(x[y == 1])
    # Majority Set

    Smaj = pd.DataFrame(x[y == -1])
    # Number of Samples we need to create
    N = len(Smaj) - len(Smin)
    # Clustering Parameters
    k1 = 5
    k2 = 3
    k3 = int(len(Smin) / 2)

    # Other Parameters
    Cp = 3
    CMAX = 2
    Cf_th = 5

    #########################---Helper Functions--################

    def closeness_factor(Yi, Xj):
        if Smin.index.get_loc(Xj.name) not in Nmin[Sbmaj.index.get_loc(Yi.name)]:
            closeness_factor = 0
        else:
            d = np.linalg.norm(Yi.values - Xj.values)
            if (1 / d) <= Cf_th:
                f = (1 / d)
            else:
                f = Cf_th
            closeness_factor = (f / Cf_th) * CMAX
        return closeness_factor

    def Davg(SminF):
        Davg = 0
        for DXindex, Dx in SminF.iterrows():
            temp = []
            for DYindex, Dy in SminF.iterrows():
                if (DYindex != DXindex):
                    temp.append(np.linalg.norm(Dy - Dx))

            Davg += min(temp)

        Davg /= len(SminF)
        return Davg

    #########################---Main Algorithm--################
    print("Setup Beginning")

    # For each minority example Xi 2 Smin, compute the nearest neighbor set, which consists of the
    # nearest k1 neighbors of Xi according to euclidean distance.

    construction = NearestNeighbors(n_neighbors=k1 + 1)
    construction.fit(pd.DataFrame(x))
    NN = construction.kneighbors(X=Smin.values, n_neighbors=k1 + 1, return_distance=False)

    # Construct the filtered minority set, Sminf by removing those minority class samples which have no minority
    # example in their neighborhood
    SFIDX = []
    for mcs in NN:
        # Get index of each minority class sample in NN and remove it from its subset (so it can't match to itself)
        tempidx = mcs[0]
        mcs = np.delete(mcs, 0)
        # Check if there are minority class samples
        if (any(Smin.index.isin(mcs)) == True):
            SFIDX.append(tempidx)

    # Construct set from indexes
    SminF = pd.DataFrame(x).loc[np.array(SFIDX)]

    # For each in SminF construct the nearest majority set (Nmaj) which consists of the nearest k2 majority
    # samples from xi according to euclidean distance.
    majority_set = NearestNeighbors(n_neighbors=k2 + 1)
    majority_set.fit(Smaj)
    Nmaj = majority_set.kneighbors(X=SminF.values, n_neighbors=k2 + 1, return_distance=False)

    # Find the borderline majority set (Sbmaj) as the union of all Nmaj sets
    Sbmaj = pd.DataFrame(x).loc[np.unique(Nmaj.reshape(-1))]

    # For each majority example in Sbmaj, compute the nearest minority set Nmin(Yi) which consists of
    # the nearest k3 minority examples from Yi by euclidean distance
    minority_set = NearestNeighbors(n_neighbors=k3 + 1)
    minority_set.fit(Smin)
    Nmin = minority_set.kneighbors(X=Sbmaj.values, n_neighbors=k3 + 1, return_distance=False)

    # Find the informative minority set, Simin, as the union of all Nmin(Yi)s
    Simin = Smin.iloc[np.unique(Nmin.reshape(-1))]

    # For each Yi belonging to Sbmaj and for each Xi belonging 2 Simin, compute the
    # information weight

    # Use this to calculate selection weights and probabilities for each Xi in Simin
    selection_weights = []
    for j, Xj in Simin.iterrows():
        weights = []
        c_f = []
        d_f = []
        for i, Yi in Sbmaj.iterrows():
            c_f.append(closeness_factor(Yi, Xj))
        for counter in range(0, len(Sbmaj)):
            density_factor = c_f[counter] / np.array(c_f).sum()
            d_f.append(density_factor)

        information_weight = np.array(c_f) * np.array(d_f)

        selection_weights.append(information_weight.sum())

    selection_probs = selection_weights / np.array(selection_weights).sum()

    # Initiliase Somin the oversampled minority set
    oversampled_Smin = Smin
    # Calculate Th the parameter we use as distance threshold in our Agglomerative Clustering
    D_avg = Davg(SminF)
    Th = D_avg * Cp

    # Seperate minority set into clusters using Agglomerative Clustering
    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='average', distance_threshold=Th)
    clusters = model.fit_predict(Smin)

    print("Sampling Beginning")
    print(str(N) + " samples to be completed...")

    # Keep track of the end of the dataframe
    loop_index = len(pd.DataFrame(x))
    # Iterate through the number of samples we need to create
    for k in range(0, N):
        # Progress Log
        if (k % 10000) == 0: print("Samples completed: " + str(k))

        # Choose a random element of Simin based on the previously calculated selection probabilities
        sample = np.random.choice(Simin.index.values, p=selection_probs)

        # Locate the sample and its cluster
        searchval = clusters[Smin.index.get_loc(sample)]
        cluster_population = np.array(np.where(clusters == searchval)).reshape(-1)

        # Select another Sample from the same cluster as our original sample
        x_sample = Smin.loc[sample].values
        y_sample = Smin.iloc[np.random.choice(cluster_population)].values

        # Calculate a random alpha between [0,1]
        alpha = np.random.uniform()
        # Create synthetic sample where s = x + alpha*(y-x)
        synthetic = x_sample + alpha * (y_sample - x_sample)

        # Append synthetic sample to the oversampled dataframe
        output = pd.Series(data=synthetic, index=pd.DataFrame(x).columns, name=loop_index)
        oversampled_Smin = oversampled_Smin.append(output)
        loop_index += 1

    # Add to original majority set
    last_y = len(Smaj) * [-1]
    last_y.extend(len(oversampled_Smin) * [1])
    result = Smaj.append(oversampled_Smin)

    res = np.array(result)
    return res, last_y


def show_plot(train_k, test_k):
    x = train_k.x
    y = train_k.y
    x_c = train_k.x_c
    y_c = train_k.y_c
    x_c = np.array(x_c)
    y_c = np.array(y_c)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c="g", s=2)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], c="r", s=2)
    plt.scatter(x_c[y_c == 1][:, 0], x_c[y_c == 1][:, 1], c="g", marker="+", s=100)
    plt.scatter(x_c[y_c == 0][:, 0], x_c[y_c == 0][:, 1], c="r", marker="+", s=100)
    plt.show()


def data_detail(file):
    # f = open('data_detail_1120_晚上_acc.json', 'r')#读取文件
    # f = open('标准化-gm-1122.json', 'r')#读取文件data_detail_1115_stander_f1
    f = open(file, 'r')
    a = f.read()
    dict_hi = eval(a)
    f.close()
    return dict_hi


def GetAucByLabel(y_true, y_pred):
    # zly 自己写的计算方式
    from sklearn import metrics
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    auc = (TPR + (1 - FPR)) / 2
    return auc

# Potential Anchoring for imbalanced data classification
# 这篇文章里面计算DI的方式
def difficulty_index(X, y, k=5, majority_class=-1):
    knn = NearestNeighbors(k + 1).fit(X)

    difficulty_coefficients = []

    for X_i, y_i in zip(X, y):
        if y_i == majority_class:
            continue
        else:
            indices = knn.kneighbors([X_i], return_distance=False)[0, 1:]
            n_majority_neighbors = sum(y[indices] == majority_class)

            difficulty_coefficients.append(n_majority_neighbors / k)

    return np.round(np.mean(difficulty_coefficients), 3)
