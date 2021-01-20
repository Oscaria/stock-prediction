# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:07:17 2020

@author: wuzix
"""

import pandas as pd
import datetime
import os

def conform_date(string):
    for fmt in ["%Y/%m/%d", "%Y-%m-%d"]:
        try:
            return datetime.datetime.strptime(string, fmt).date()
        except ValueError:
            continue
    raise ValueError(string)


data_dir='./threshold'
data_dir2 = './threshold/afterMerge'

#ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
ths = [0.01]

for th in ths:
    print(th)

    news_file='./raw data/中兴新闻矩阵_3.csv'
    finance_file=os.path.join(data_dir,'Zhongxing财务_删0_label'+str(th)+'.csv')
    
    df=pd.read_csv(news_file,error_bad_lines=False)
    cf=pd.read_csv(finance_file,error_bad_lines=False)
    
    news_date=df.ix[:,0]
    #df=pd.DataFrame(index=news_date)
    finance_date=cf.ix[:,1]
    
    '''
    for i in range(len(news_date)):
        news_date[i] = conform_date(news_date[i])
        #print(news_date[i])
    
        
    for i in range(len(finance_date)):
        finance_date[i] = conform_date(finance_date[i])
        
    '''
    
    print(news_date[0])
    print(finance_date[0])
    #differ=set(news_date).difference(finance_date)
    #print(differ)
    differ=set(news_date)^set(finance_date)
    #print(differ)
    drop=[]
    drop2=[]
    #drop.append(0)
    drop2.append(0)
    drop_date=[]
    a=[]
    
    for i in range(len(news_date)):
        if news_date[i] not in differ:
            drop.append(i)
    for i in range(len(finance_date)):
        if finance_date[i] not in differ:
            drop2.append(i+1)
    
    '''    
    with open('./raw data/中兴新闻矩阵_3.csv','r',encoding='utf-8') as r:
        lines=r.readlines()
        
        #print(lines)
        with open('./raw data/afterMerge/中兴新闻矩阵_with财务删0_afterMerge_5.csv','w',encoding='utf-8') as w:
           for i in range(len(drop)):
                w.write(lines[drop[i]]) 
    '''
    
    
           
                
    with open(finance_file,'r',encoding='utf-8') as r:
        lines=r.readlines()
        
        #print(lines)
        fname2 = os.path.join(data_dir2,'Zhongxing财务_删0_label'+str(th)+'_forVisual_afterMerge.csv')
        with open(fname2,'w',encoding='utf-8') as w:
           for i in range(len(drop2)):
                w.write(lines[drop2[i]])
    
