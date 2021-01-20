# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:21:12 2020

@author: AAA
"""
import pandas as pd
import numpy as np
import os

df = pd.read_csv('./raw data/csv/Zhongxing财务_删0.csv')
df['label'] = 0
print(df.iloc[0, 8])

data_dir='./threshold'
ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]

for th in ths:
    print(th)
    for i in range(len(df)):
        if df.iloc[i, 8] >= th:
            df.iloc[i, 14] = 1
        else:
            df.iloc[i, 14] = 0
    fname=os.path.join(data_dir,'Zhongxing财务_删0_label'+str(th)+'.csv')
    df.to_csv(fname, index = False)
