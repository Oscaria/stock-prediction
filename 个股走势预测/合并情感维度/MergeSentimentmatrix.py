#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:22:38 2020

@author: xuyuemei

合并新闻的情感值
"""

import pandas as pd
import numpy as np

df = pd.read_csv('./afterMerge/Zhongxing_predict_mergeNews-1_with财务删0_afterMerge.csv', header = None, encoding = 'utf-8')
#df2= pd.DataFrame()
time=df.iloc[:,0]
predict=df.iloc[:,1]
resultTime=[]
resultSentiment=np.zeros(len(set(time))-1).astype(int)
count=np.zeros(len(set(time))-1)
j=-1

for i in range(1,len(time)):
    if(time[i]!=time[i-1]): 
        resultTime.append(time[i])
        j=j+1
        #print("@@@", type(resultSentiment[j]))
        #print('###', type(predict[i]))
        resultSentiment[j] += int(predict[i])
        count[j]+=1
    else:
        resultSentiment[j] += int(predict[i])
        count[j] += 1

df2=pd.DataFrame()
        
df2.loc[:,0]=resultTime
df2.loc[:,1]=resultSentiment
#df2.loc[:,1]=np.asarray(resultSentiment)/np.asarray(count,dtype=float)
df2.to_csv(r'./sentiMerge/Zhongxing_predict_mergeNews-1_with财务删0_afterMerge_sentiMerge.csv',index=False)
        
        
    
