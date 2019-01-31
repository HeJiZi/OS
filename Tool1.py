# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:32:47 2019

@author: SmacUL
"""

import pandas as pd
import numpy as np
import os

def get_detail(labs):
    
    detail = ""
    
    fi_lab = ""
    se_lab = ""
    th_lab = ""
    
    fi_cou = 1
    se_cou = 1
    th_cou = 1
    
    for i, lab in enumerate(labs):
        
        lev_lab = lab.split("--")[0]

#        lev_lab = lab.split("--")[0]
#        if fi_lab != lev_lab:
#            if i != 0:
#                # write into file
#                detail = detail + fi_lab + "\t" + str(fi_cou) + "\n"
#            fi_lab = lev_lab
#            fi_cou = 1
#        else:
#            fi_cou += 1
#        
#        
#        lev_lab = lab.split("--")[1]
#        if se_lab != lev_lab:
#            if i != 0:
#                # write into file
#                detail = detail + "\t" + se_lab + "\t" + str(se_cou) + "\n"
#            se_lab = lev_lab
#            se_cou = 1
#        else:
#            se_cou += 1
#            
#        lev_lab = lab.split("--")[2]
#        if th_lab != lev_lab:
#            if i != 0:
#                # write into file
#                detail = detail + "\t\t" + th_lab + "\t" + str(th_cou) + "\n"
#            th_lab = lev_lab
#            th_cou = 1
#        else:
#            th_cou += 1
    
    with open("./detail.txt", 'w') as f:
        f.write(detail)


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
    
    labs = ori_data[:, 1]
    
    
    get_detail(labs)