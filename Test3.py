# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:01:55 2019

提供了一个简易的处理流程。
通过学习 12000 条商品信息来预测商品的一级标签。

@author: SmacUL
"""

import pandas as pd
import numpy as np
import jieba
import spacy

from keras.models import load_model
from keras import layers
from keras import models
from keras.utils import to_categorical


if __name__ == "__main__":
    
    tra_ori_path = "./data/train.tsv"
    ori_code = "gb18030"
    
    # 读取数据
    ori_data_df = pd.read_csv(tra_ori_path, 
                               sep='\t',
                               encoding=ori_code,
                               nrows=None
                               )
    
    # 转 DataFrame 为 NdArray
    ori_data = np.array(ori_data_df)
    
    # 打乱数据
    np.random.shuffle(ori_data)
    
    group_size = 20000
    perc = 0.6
    
    sams = ori_data[0: group_size, 0]
    labs = ori_data[0: group_size, 1]
    
    tra_sams = sams[0: int(perc * len(sams))]
    tra_labs = labs[0: int(perc * len(labs))]
    
    tes_sams = sams[int(perc * len(sams)): len(sams)]
    tes_labs = labs[int(perc * len(labs)): len(labs)]
    
    print("开始分词")
    
    tra_toks = []
    for tra_sam in tra_sams:
        toks = jieba.cut(tra_sam)
        tra_toks.append(toks)
    tra_toks = np.array(tra_toks)
    tes_toks = []
    for tes_sam in tes_sams:
        toks = jieba.cut(tes_sam)
        tes_toks.append(toks)
    tes_toks = np.array(tes_toks)
    
    print("分词完成")
    print("开始生成词向量")
    
    nlp = spacy.load("zh")
    
    tra_charas = np.zeros([len(tra_toks), 20, 128])
    for i, tokens in enumerate(tra_toks):
        for j, token in enumerate(tokens):
            if j == 20 or type(token) == float:
                break
            tra_charas[i][j] = nlp.vocab[token].vector
            
    tes_charas = np.zeros([len(tes_toks), 20, 128])
    for i, tokens in enumerate(tes_toks):
        for j, token in enumerate(tokens):
            if j == 20 or type(token) == float:
                break
            tes_charas[i][j] = nlp.vocab[token].vector

    print("词向量生成完毕")

    type_dict = {'本地生活': 0, '宠物生活': 1, '厨具锅具': 2, '电脑/办公': 3, 
                 '服饰鞋帽': 4, '家居家装': 5, '家用/商用家具': 6, '家用电器': 7, 
                 '家装建材': 8, '教育音像': 9, '母婴用品/玩具乐器': 10, 
                 '汽配用品': 11, '生鲜水果': 12, '食品/饮料/酒水': 13, 
                 '手机数码': 14, '图书杂志': 15, '箱包皮具': 16, '医药保健': 17,
                 '音乐影视': 18, '运动户外': 19, '钟表礼品': 20, '珠宝饰品': 21}
    
    print("开始处理标签")
    
    tra_y = []
    for tra_lab in tra_labs:
        tra_y.append(type_dict[tra_lab.split('--')[0]])
    tra_y = np.array(tra_y)
    tra_y = to_categorical(tra_y)

    tes_y = []
    for tes_lab in tes_labs:
        tes_y.append(type_dict[tes_lab.split('--')[0]])
    tes_y = np.array(tes_y)
    tes_y = to_categorical(tes_y)
    
    print("标签处理完成")
    
    model = models.Sequential()
    
    model.add(layers.Conv1D(32, 4, activation='relu', input_shape=(20, 128)))
    model.add(layers.Conv1D(64, 4, activation='relu'))
    model.add(layers.Conv1D(64, 4, activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(22, activation='softmax'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(tra_charas, tra_y, epochs=5, batch_size=512)
    test_loss, test_acc = model.evaluate(tes_charas, tes_y, batch_size=512)
    print("test loss is : ", test_loss)
    print("test accuracy is : ", test_acc)
    model.save('my_model.h5')

    print("开始处理验证数据")
    
    # 验证集数据处理
    val_sams = ori_data[30000: 30004, 0]
    val_labs = ori_data[30000: 30004, 1]
    print(val_sams)
    print(val_labs)
    val_toks = []
    for val_sam in val_sams:
        toks = jieba.cut(val_sam)
        val_toks.append(toks)
    vak_toks = np.array(val_toks)
    
    val_charas = np.zeros([len(val_toks), 20, 128])
    for i, tokens in enumerate(val_toks):
        for j, token in enumerate(tokens):
            if j == 20 or type(token) == float:
                break
            val_charas[i][j] = nlp.vocab[token].vector
    
    print("验证数据处理完成")
    
    res = model.predict(val_charas)

    for i in range(len(res)):
        print("预测的各个种类的概率：")
        print(res[i])
        print("最大概率为：", max(res[i]))
        for key, value in enumerate(res[i]):
            if max(res[i]) == value:
                print("商品种类下标: ", key)
        print()
