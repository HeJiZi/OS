#!/usr/bin/env python
# encoding: utf-8
"""
Created on 2019/1/31 17:46

针对数据标签处理

    get_detail 方法
        可以给出所有标签数据的层次关系，以及标签对应的商品数量和位置信息
    get_lab_dict_by_name 方法
        通过给出一个父级标签，此方法将获取父级标签下所有的子标签，
        并将子标签转化为数字，提供两种不用索引的字典
    get_range_by_name 方法
        通过给出的标签名称，计算出标签对应的商品信息下标范围，闭区间

@author: SmacUL
"""

def get_detail(labs):
    """ 生成所有级别的商品标签以及相应的数量和位置信息

    生成的内容用于数据分析

    输出格式

        0. 本地生活 350 [0, 349]
            0. 游戏充值 350 [0, 349]
                    0. QQ充值 41 [0, 40]		1. 游戏点卡 309 [41, 349]

        1. 宠物生活 2268 [350, 2617]
            0. 宠物零食 64 [350, 413]
                    0. 猫零食 19 [350, 368]		1. 磨牙/洁齿 45 [369, 413]
                        ··· ···

    :param labs:    按序的完整的商品标签数据
    :return detail: 所有级别的商品标签以及相应的数量和位置信息
    """

    detail = ""

    fi_s = -1
    fi_e = -1
    fi_cou = -1
    while fi_e != len(labs) -1:

        detail = detail + '\n'

        fi_s = fi_e + 1
        fi_lab_bas = labs[fi_s].split('--')[0]
        for i in range(fi_s, len(labs)):
            fi_lab = labs[i].split('--')[0]
            if fi_lab != fi_lab_bas:
                fi_e = i - 1
                fi_cou += 1
                detail = detail + str(fi_cou) + '. ' + fi_lab_bas + ' ' + str(fi_e - fi_s + 1) + \
                         ' [' + str(fi_s) + ', ' +  str(fi_e) + ']\n'
                break
            if i == len(labs) - 1:
                fi_e = i
                fi_cou += 1
                detail = detail + str(fi_cou) + '. ' + fi_lab_bas + ' ' + str(fi_e - fi_s + 1) + \
                         ' [' + str(fi_s) + ', ' +  str(fi_e) + ']\n'
                break

        se_s = fi_s - 1
        se_e = fi_s - 1
        se_cou = -1
        while se_e != fi_e:
            se_s = se_e + 1
            se_lab_bas = labs[se_s].split('--')[1]
            for j in range(se_s, len(labs)):
                se_lab = labs[j].split('--')[1]
                if se_lab != se_lab_bas:
                    se_e = j - 1
                    se_cou += 1
                    detail = detail + '\t' + str(se_cou) + '. ' + se_lab_bas + ' ' + str(se_e - se_s + 1) + \
                             ' [' + str(se_s) + ', ' +  str(se_e) + ']\n'
                    break
                if j == len(labs) - 1:
                    se_e = j
                    se_cou += 1
                    detail = detail + '\t' + str(se_cou) + '. ' + se_lab_bas + ' ' + str(se_e - se_s + 1) + \
                             ' [' + str(se_s) + ', ' +  str(se_e) + ']\n'
                    break

            detail = detail + '\t\t\t'

            th_s = se_s - 1
            th_e = se_s - 1
            th_cou = -1
            while th_e != se_e:
                th_s = th_e + 1
                th_lab_bas = labs[th_s].split('--')[2]
                for k in range(th_s, len(labs)):
                    th_lab = labs[k].split('--')[2]
                    if th_lab != th_lab_bas:
                        th_e = k - 1
                        th_cou += 1
                        detail = detail + str(th_cou) + '. ' + th_lab_bas + ' ' + str(th_e - th_s + 1) + \
                                 ' [' + str(th_s) + ', ' +  str(th_e) + ']\t\t'
                        if (th_cou + 1) % 2 == 0:
                            detail += '\n\t\t\t'
                        break
                    if k == len(labs) - 1:
                        th_e = k
                        th_cou += 1
                        detail = detail + str(th_cou) + '. ' + th_lab_bas + ' ' + str(th_e - th_s + 1) + \
                                 ' [' + str(th_s) + ', ' +  str(th_e) + ']\t\t'
                        break

            detail = detail + '\n'

    return detail


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
    # 获得 labs 数据
    labs = ori_data[:, 1]


    # get_detail 测试代码

    # detail = get_detail(labs)
    # with open('./detail.txt', 'w', encoding='utf-8') as f:
    #     f.write(detail)



    # get_lab_dict_by_name 测试代码

    # dict_lab, dict_num = get_lab_dict_by_name(labs, '宠物生活', 1)
    # print(dict_lab)
    # print(dict_num)
    # dict_lab, dict_num = get_lab_dict_by_name(labs)
    # print(dict_lab)
    # print(dict_num)
