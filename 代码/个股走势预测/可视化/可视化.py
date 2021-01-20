# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:56:02 2020

@author: wuzix
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from matplotlib import rc
from pylab import *  
from matplotlib.pyplot import MultipleLocator
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  

file='./test_visual_2.csv'
df=pd.read_csv(file,error_bad_lines=False,encoding='utf-8')
df.columns=['DATE','CLOSE','PREDICT']
date=[]
close=[]
predict=[]
for i in range(len(df['DATE'])//5):
    date.append(df['DATE'][i])
    close.append(df['CLOSE'][i])
    predict.append(df['PREDICT'][i])

figure=plt.figure(figsize=(15,7))
#figsize = 15,7
#figure=plt.subplots(figsize=figsize)
ax = figure.add_subplot(111)
plt.ylim(0,40)
ax.plot(date,close, color='b',linewidth=1.5,label='Actual Closing Price')
ax.set_xticklabels(date, rotation=90)
ax.scatter(date,predict,color='r',marker='X',label='Predicted Closing Price')
plt.xlabel('Dates', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)
plt.title('Stock Code:000063.SZ', fontsize=10)
figure.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

plt.show()