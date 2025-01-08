import copy
from collections import Counter

from numpy import NAN
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score as G_mean
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score

from start import Cssc, Kmean_Test, suppress_stdout_stderr, distance
def formal_CSSC_prod(test_k):
    Confidence = []
    max_n = max(test_k.u1)
    min_n = min(test_k.u1)
    for i in range(len(test_k.y)):
        if test_k.c1[i] == -1:
            curr_confidence = 0.5 * (test_k.u1[i] - min_n) / (max_n - min_n) + 0.5
            Confidence.append([curr_confidence, 1 - curr_confidence])
        else:
            curr_confidence = 0.5 * (test_k.u1[i] - min_n) / (max_n - min_n) + 0.5
            Confidence.append([1 - curr_confidence, curr_confidence])
        # print("{}\t{}\t{}".format(test_k.c1[i],Confidence[-1][0],Confidence[-1][1]))
    test_k.kmeans_confidence = Confidence

def afterPrepoocessLabel1(Com_ori, Com_pred, Com_confidence, Comdata, CSSC, test_k, K_choose=10, lamba_one=0.2):
    # 获得对比实验的结果
    # print("初始算法\nAUC:{}\tGM:{}\tF1:{}".format(first_AUC, first_GM, first_F1))
    # 初始算法的性能
    train_y = CSSC.y

    Neg_all, Neg_acc, Pos_all, Pos_acc = 0, 0, 0, 0
    for i in range(len(Com_ori)):
        if Com_ori[i] == -1:
            Neg_all += 1
            if Com_ori[i] == Com_pred[i]:
                Neg_acc += 1
        if Com_ori[i] == 1:
            Pos_all += 1
            if Com_ori[i] == Com_pred[i]:
                Pos_acc += 1
    print("Pos的ACC:{}({}/{})\nNeg的ACC:{}({}/{})".format(Pos_acc / Pos_all, Pos_acc, Pos_all, Neg_acc / Neg_all,
                                                          Neg_acc, Neg_all))

    TrainPos, TrainNeg, PredPos, PredNeg = Counter(train_y)[1], Counter(train_y)[-1], Counter(Com_pred)[1], \
                                           Counter(Com_pred)[-1]

    print("\nTrainPos:{}\nTrainNeg:{}\nPredPos:{}\nPredNeg:{}\n".format(Counter(train_y)[1], Counter(train_y)[-1],
                                                                        Counter(Com_pred)[1], Counter(Com_pred)[-1]))
    # 对比算法和Kmeans在train上的f1_score
    Com_f1 = Comdata[-1]
    K_f1 = CSSC.train_f1
    # 两个分类器的权重权重
    weight = Com_f1 / (Com_f1 + K_f1)
    print("CompareAlg:{}\tCSSC:{}".format(Com_f1, K_f1))
    print("权重:{}".format(weight))
    # 记录Train以及Pred上的IR
    TrainIR = TrainNeg / TrainPos
    if PredPos == 0:
        PredPos = 0.01
    PredIR = PredNeg / PredPos
    print("TrainIR:{}\tPredIR:{}".format(TrainIR, PredIR))
    if TrainIR * 0.9 < PredIR < TrainIR * 1.1:
        print("该数据集不用处理")
        return Com_pred, NAN
    changeFlag = NAN
    changeNum = NAN
    # Neg多数类预测多了
    if PredIR > TrainIR * 1.1:
        changeFlag = "Neg to Pos"
        changeNum = (PredNeg - TrainIR * PredPos) / (1 + TrainIR)
    # Neg多数类预测少了
    if PredIR < TrainIR * 0.9:
        changeFlag = "Pos to Neg"
        changeNum = (TrainIR * PredPos - PredNeg) / (1 + TrainIR)
    print("标识", changeFlag, changeNum)
    # 在CSSC上，由于是不平衡分类，概率预测需要做出一点转换
    formal_CSSC_prod(test_k)
    # 开始更改标签
    # 存放所有需要更改的标签
    data_flag = []
    if changeFlag == "Pos to Neg":
        # 记录所有可能需要更改的样本
        for i in range(len(Com_pred)):
            if Com_pred[i] == 1:
                item = {}
                # 记录对应的数据index
                item["index"] = i
                item["real"] = test_k.y[i]
                item["ComPred"] = Com_pred[i]
                item["CSSCPred"] = test_k.c1[i]
                # 记录对应的数据的原始算法的在Neg的置信度
                item["Comconfidence"] = Com_confidence[i][0]
                # 记录对应的kmeans上Neg的置信度
                item["Kmeansconfidence"] = test_k.kmeans_confidence[i][0]
                # 记录融合的概率
                item["confidence"] = weight * Com_confidence[i][0] + (1 - weight) * test_k.kmeans_confidence[i][0]
                data_flag.append(item)
    else:
        # 记录所有可能需要更改的样本
        for i in range(len(Com_pred)):
            if Com_pred[i] == -1:
                item = {}
                # 记录对应的数据index
                item["index"] = i
                item["real"] = test_k.y[i]
                item["ComPred"] = Com_pred[i]
                item["CSSCPred"] = test_k.c1[i]
                # 记录对应的数据的原始算法的在Neg的置信度
                item["Comconfidence"] = Com_confidence[i][1]
                # 记录对应的kmeans上Neg的置信度
                item["Kmeansconfidence"] = test_k.kmeans_confidence[i][1]
                # 记录融合的概率
                item["confidence"] = weight * Com_confidence[i][1] + (1 - weight) * test_k.kmeans_confidence[i][1]
                data_flag.append(item)
    haveChanged = 0
    # if Com_f1 >= K_f1:
    #     data_flag = sorted(data_flag, key=lambda i: i["Comconfidence"], reverse=True)
    # else:
    #     data_flag = sorted(data_flag, key=lambda i: i["Kmeansconfidence"], reverse=True)
    data_flag = sorted(data_flag, key=lambda i: i["confidence"], reverse=True)
    data_flag = data_flag[:int(len(data_flag) / 2)]
    # 开始修改标签 # Pos2Neg
    changed_acc = 0
    if changeFlag == "Pos to Neg":
        # 这边计算我们挑选的数据集周围是否大多数是多数类
        for item in data_flag:
            if item["confidence"] < 0.5:
                continue
            # 获得周围邻居的信息
            nei_list = []
            for train_index in range(len(CSSC.y)):
                nei = {}
                curr_test = test_k.x[item["index"]]
                nei["train_index"] = train_index
                nei["train_label"] = CSSC.y[train_index]
                nei["distance"] = distance(curr_test, CSSC.x[train_index])
                nei_list.append(nei)
            nei_list = sorted(nei_list, key=lambda i: i["distance"], reverse=False)
            nei_list = nei_list[0: K_choose]
            differ_samper = 0
            same_samper = 0

            for i in nei_list:
                if item["ComPred"] != i["train_label"]:
                    differ_samper += 1
                else:
                    same_samper += 1
            # print(differ_samper, same_samper)
            # 计算周围最近的几个样本的标签得到我们是否需要修改这个样本
            if haveChanged < changeNum * (1 - lamba_one) and differ_samper / (K_choose) > 0.80:
                # print("{}修改一个样本：({}/{})".format(changeFlag,differ_samper,K_choose))
                haveChanged += 1
                Com_pred[item["index"]] = -1
                if Com_pred[item["index"]] == item["real"]:
                    changed_acc += 1
    else:  # Neg2Pos
        for item in data_flag:
            # 获得周围邻居的信息
            nei_list = []
            for train_index in range(len(CSSC.y)):
                nei = {}
                curr_test = test_k.x[item["index"]]
                nei["train_index"] = train_index
                nei["train_label"] = CSSC.y[train_index]
                nei["distance"] = distance(curr_test, CSSC.x[train_index])
                nei_list.append(nei)
            nei_list = sorted(nei_list, key=lambda i: i["distance"], reverse=False)
            nei_list = nei_list[0: K_choose]
            differ_samper = 0
            same_samper = 0
            for i in nei_list:
                if item["ComPred"] != i["train_label"]:
                    differ_samper += 1
                else:
                    same_samper += 1
            if haveChanged < changeNum * (1 + lamba_one) and differ_samper / (K_choose) > 0.3:  # 0.3
                # print("{}修改一个样本：({}/{})".format(changeFlag, differ_samper, K_choose))
                haveChanged += 1
                Com_pred[item["index"]] = 1
                if Com_pred[item["index"]] == Com_ori[item["index"]]:
                    changed_acc += 1
    print("理论修改的样本数量:{}".format(changeNum))
    print("修改的样本个数:{}".format(haveChanged))
    # 最终话算法的性能
    Neg_all, Neg_acc, Pos_all, Pos_acc = 0, 0, 0, 0
    for i in range(len(Com_ori)):
        if Com_ori[i] == -1:
            Neg_all += 1
            if Com_ori[i] == Com_pred[i]:
                Neg_acc += 1
        if Com_ori[i] == 1:
            Pos_all += 1
            if Com_ori[i] == Com_pred[i]:
                Pos_acc += 1
    print("\nPos的ACC:{}({}/{})\nNeg的ACC:{}({}/{})".format(Pos_acc / Pos_all, Pos_acc, Pos_all, Neg_acc / Neg_all,
                                                            Neg_acc, Neg_all))
    # 保留部分的detail信息
    res = {}
    res["changeFlag"] = changeFlag
    res["changeNum"] = changeNum
    res["haveChanged"] = haveChanged
    res["changed_acc"] = changed_acc
    return Com_pred, res

def PPF(train_x, train_y, test_x, test_y, model = SVC()):
    ori_train_x = copy.deepcopy(train_x)
    ori_train_y = copy.deepcopy(train_y)
    #
    clf = model
    clf.fit(train_x, train_y)
    y_pred_ori = clf.predict(test_x)
    y_pred_prob = clf.predict_proba(test_x)
    # decis = clf.decision_function(test_x)

    # zly_prob_jun = jun(decis)

    train_pred = clf.predict(ori_train_x)
    train_gm = G_mean(ori_train_y, train_pred)
    train_f1 = f1_score(ori_train_y, train_pred, labels=[1])
    train_pred_pro = clf.predict_proba(ori_train_x)
    # train_jun = jun(clf.decision_function(ori_train_x))
    other_this = dict(Counter(train_y))

    # 聚类部分
    CSSC = Cssc(maxnum=20, weight=0.3)
    CSSC.fit(train_x, train_y)
    test_k = Kmean_Test(test_x, test_y)
    # 屏蔽聚类中间过程的print
    with suppress_stdout_stderr():
        # 初始化质心的过程
        CSSC.k_center()
        # 质心迭代
        CSSC.kmean_train(test_k)
        # 得到聚类结束后的结果
        train_pred = CSSC.kmean_predict0()
        test_pred = CSSC.kmean_predict_testdata0(test_k)
        # 聚类结束

    Com_pred, res = afterPrepoocessLabel1(test_y, y_pred_ori, y_pred_prob,
                                          [other_this[-1], other_this[1], train_gm, train_f1],
                                          CSSC, test_k, K_choose=5)
    return Com_pred
