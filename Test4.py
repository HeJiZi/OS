# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:44:15 2019

在 Test3 的基础上做出了修改，使其能够一次性处理 50 万的数据，
对第一级标签进行预测。


    8G 的内存可以一次性处理，当然内存更大，肯定更好。
    生成词向量那一块是最消耗内存的，
    感觉 generator 这种数据类型帮了忙。

    生成词向量之后，尝试对 tra_charas 和 tes_charas 中的数据类型进行修改，
    例如将 float64 调整为 float32 ，
    这些操作很容易引起 Memory Error ，应该还是内存不足的锅。

@author: SmacUL
"""

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
    ori_data_df = pd.read_csv(tra_ori_path, 
                               sep='\t',
                               encoding=ori_code,
                               nrows=None
                               )
    
    # 转 DataFrame 为 NdArray
    ori_data = np.array(ori_data_df)
    
    # 打乱数据
    np.random.shuffle(ori_data)
    
    perc = 0.6
    
    sams = ori_data[:, 0]
    labs = ori_data[:, 1]
    
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
    
    input_shape = (20, 128)
    batch_size = 512
    epochs = 5

    model = models.Sequential()
    
    model.add(layers.Conv1D(32, 4, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(64, 4, activation='relu'))
    model.add(layers.Conv1D(64, 4, activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(22, activation='softmax'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(tra_charas, tra_y, epochs=epochs, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(tes_charas, tes_y, batch_size=batch_size)
    print("test loss is : ", test_loss)
    print("test accuracy is : ", test_acc)

    model_name = '1-' + str(test_acc)[2:]
    model_path = './model'
    if not os.path.exists(model_path):
        os.path.makedirs(model_path)
    model.save(os.path.join(model_path, model_name + '.h5'))


    