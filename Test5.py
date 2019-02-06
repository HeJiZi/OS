#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
Created on 02/06/2019 20:31 

对 1 级标签（ 0 开始计数）进行分类训练

@author: SmacUL 
"""


def get_lab_dict_by_name(labs, name=None, lev=0):
    """ 通过给出的父级标签名称，生成对应的子级标签的字典

    例如给出的 name 为 宠物生活 ， lev 为 1 （由 0 开始计数）

        宠物生活
            宠物零食    宠物玩具    宠物主粮    出行装备    家居日用    洗护美容    医疗保健

    输出为

        dict_lab:
            {'宠物零食': 0, '宠物玩具': 1, '宠物主粮': 2, '出行装备': 3,
             '家居日用': 4, '洗护美容': 5, '医疗保健': 6}
        dict_num:
            {0: '宠物零食', 1: '宠物玩具', 2: '宠物主粮', 3: '出行装备',
            4: '家居日用', 5: '洗护美容', 6: '医疗保健'}

    如果 lev 为 1 或 2 以外的数字，此方法将给出第 0 级商品标签

    :param labs:    按序的完整的商品标签数据
    :param name:    父级标签的名称， lev 参数为 1 或 2 时有效
    :param lev:     需要生成字典的商品标签的级别，即子级标签的级别，默认为 0 ，输出第 0 级的商品标签
    :return dict_lab:   标签--数字 字典，以标签为索引
    :return dict_num:   数字--标签 字典，以数字为索引
    """

    ls = []
    if lev == 1 or lev == 2:
        print("当前生成第 %d 级标签字典（由 0 开始计数）" % lev)
        for lab in labs:
            par_lab = lab.split('--')[lev - 1]
            if name == par_lab:
                lev_lab = lab.split('--')[lev]
                if lev_lab not in ls:
                    ls.append(lev_lab)
    else:
        print("当前生成第 0 级标签字典（由 0 开始计数）")
        for lab in labs:
            lev_lab = lab.split('--')[0]
            if lev_lab not in ls:
                ls.append(lev_lab)

    dict_lab = {}
    dict_num = {}
    for num, lab in enumerate(ls):
        dict_lab[lab] = num
        dict_num[num] = lab

    return dict_lab, dict_num


def get_range_by_name(labs, name=None, lev=0):
    """ 通过给出的标签名称，计算出标签对应的商品信息下标范围

    For example

        给出标签名称 宠物生活 ，返回闭区间 [350, 2617]

    如果 lev 的值为 0 ，那么方法将返回闭区间 [0, len(labs) - 1]

    :param labs: 按序的完整的商品标签数据
    :param name: 标签的名称， lev 参数为 1 或 2 时有效
    :param lev: 需要生成范围的商品标签的级别，默认为 0 ，输出第 0 级的商品标签
    :return range: 对应商品的下标范围，闭区间
    """
    range = []
    last = -1
    if lev == 1 or lev == 2:
        for i, lab in enumerate(labs):
            par_lab = lab.split('--')[lev - 1]
            if name == par_lab:
                if last == -1:
                    range.append(i)
                last = i
        range.append(last)
    else:
        range.append(0)
        range.append(len(labs) - 1)

    return range


import pandas as pd
import numpy as np
import jieba
import spacy
import os

from keras.models import load_model
from keras import layers
from keras import models
from keras.utils import to_categorical

if __name__ == "__main__":

    tra_ori_path = "./data/train.tsv"
    ori_code = "gb18030"

    # 读取数据
    ori_data_df = pd.read_csv(tra_ori_path, sep='\t', encoding=ori_code, nrows=None)

    # 转 DataFrame 为 NdArray
    ori_data = np.array(ori_data_df)

    labs = ori_data[:, 1]

    fi_lab_dict, fi_num_dict = get_lab_dict_by_name(labs)
    perc = 0.7
    lev = 1
    # group running
    for index, value in enumerate(fi_lab_dict):
        print("start handle " + str(index) + " " + value)

        lev_lab_dict, lev_num_dict = get_lab_dict_by_name(labs, value, lev)

        if len(lev_lab_dict) == 1:
            continue

        range = get_range_by_name(labs, value, lev)
        lev_data = ori_data[range[0]: (range[1] + 1)]
        np.random.shuffle(lev_data)
        lev_sams = lev_data[:, 0]
        lev_labs = lev_data[:, 1]

        lev_tra_sams = lev_sams[0: int(perc * len(lev_sams))]
        lev_tra_labs = lev_labs[0: int(perc * len(lev_labs))]

        lev_tes_sams = lev_sams[int(perc * len(lev_sams)): len(lev_sams)]
        lev_tes_labs = lev_labs[int(perc * len(lev_labs)): len(lev_labs)]

        print("开始分词")

        lev_tra_toks = []
        for lev_tra_sam in lev_tra_sams:
            toks = jieba.cut(lev_tra_sam)
            lev_tra_toks.append(toks)
        lev_tra_toks = np.array(lev_tra_toks)
        lev_tes_toks = []
        for lev_tes_sam in lev_tes_sams:
            toks = jieba.cut(lev_tes_sam)
            lev_tes_toks.append(toks)
        lev_tes_toks = np.array(lev_tes_toks)

        print("分词完成")

        print("开始生成词向量")

        nlp = spacy.load("zh")

        lev_tra_charas = np.zeros([len(lev_tra_toks), 20, 128])
        for i, tokens in enumerate(lev_tra_toks):
            for j, token in enumerate(tokens):
                if j == 20 or type(token) == float:
                    break
                lev_tra_charas[i][j] = nlp.vocab[token].vector

        lev_tes_charas = np.zeros([len(lev_tes_toks), 20, 128])
        for i, tokens in enumerate(lev_tes_toks):
            for j, token in enumerate(tokens):
                if j == 20 or type(token) == float:
                    break
                lev_tes_charas[i][j] = nlp.vocab[token].vector

        print("词向量生成完毕")

        print("开始处理标签")

        lev_tra_y = []
        for lev_tra_lab in lev_tra_labs:
            lev_tra_y.append(lev_lab_dict[lev_tra_lab.split('--')[lev]])
        lev_tra_y = np.array(lev_tra_y)
        lev_tra_y = to_categorical(lev_tra_y)

        lev_tes_y = []
        for lev_tes_lab in lev_tes_labs:
            lev_tes_y.append(lev_lab_dict[lev_tes_lab.split('--')[lev]])
        lev_tes_y = np.array(lev_tes_y)
        lev_tes_y = to_categorical(lev_tes_y)

        print("标签处理完成")

        input_shape = (20, 128)
        batch_size = 128
        epochs = 20

        model = models.Sequential()

        model.add(layers.Conv1D(32, 4, activation='relu', input_shape=input_shape))
        model.add(layers.Conv1D(64, 4, activation='relu'))
        model.add(layers.Conv1D(64, 4, activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(len(lev_lab_dict), activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(lev_tra_charas, lev_tra_y, epochs=epochs, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(lev_tes_charas, lev_tes_y, batch_size=batch_size)
        print("test loss is : ", test_loss)
        print("test accuracy is : ", test_acc)

        model_name = str(lev) + '-' + str(index) + '-' + str(test_acc)[2:]
        model_path = './model/' + str(lev)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(os.path.join(model_path, model_name + '.h5'))

