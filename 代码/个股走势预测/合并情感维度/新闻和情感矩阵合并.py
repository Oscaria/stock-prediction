# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:07:17 2020

@author: wuzix
"""

import pandas as pd

news_file='../建立财务矩阵/raw data/afterMerge/中兴新闻矩阵_with财务删0_afterMerge_5.csv'
finance_file='./Zhongxing_predict_merge_-1.csv'

df=pd.read_csv(news_file,error_bad_lines=False)
cf=pd.read_csv(finance_file,error_bad_lines=False)
news_date=df.ix[:,0]
#df=pd.DataFrame(index=news_date)
finance_date=cf.ix[:,0]

print(news_date[0])
print(finance_date[0])
#differ=set(news_date).difference(finance_date)
#print(differ)
differ=set(news_date)^set(finance_date)
#print(differ)
drop=[]
drop2=[]
drop.append(0)
drop2.append(0)
drop_date=[]
a=[]
for i in range(len(news_date)):
    if news_date[i] not in differ:
        drop.append(i+1)
for i in range(len(finance_date)):
    if finance_date[i] not in differ:
        drop2.append(i+1)

'''
with open('./中兴新闻矩阵.csv','r',encoding='utf-8') as r:
    lines=r.readlines()
    
    #print(lines)
    with open('./中兴新闻矩阵_afterMerge.csv','w',encoding='utf-8') as w:
       for i in range(len(drop)):
            w.write(lines[drop[i]])    

'''
       
            
with open('./Zhongxing_predict_merge_-1.csv','r',encoding='utf-8') as r:
    lines=r.readlines()
    
    #print(lines)
    with open('./afterMerge/Zhongxing_predict_mergeNews-1_with财务删0_afterMerge.csv','w',encoding='utf-8') as w:
       for i in range(len(drop2)):
            w.write(lines[drop2[i]])    
