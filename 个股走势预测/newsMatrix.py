# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:35:04 2019

@author: wuzix
"""

import pandas as pd
import numpy as np
import json
import csv


labels = json.loads(open('../新闻事件分类器/训练结果/trained_results_1594750510/labels.json').read())
#file='D:\\毕业设计\\代码\\newtry\\新闻事件预测\\格力新闻矩阵.csv'
col=[]
col.append('日期')
for i in range (len(labels)):
    col.append(labels[i])


#file='D:\\毕业设计\\代码\\newtry\\预测结果\\predicted_results_1582632892\\predictions_all_2020-03-04.csv'
#cf=pd.read_csv(file,encoding='utf-8',sep='|')
#date=cf['Date'][0]
#print(date)

#df.to_csv('D:\\毕业设计\\代码\\newtry\\新闻事件预测\\中兴新闻矩阵.csv')

with open('../新闻事件分类器/预测结果/predicted_results_1594750510/predictions_all_2020-07-15-3.csv','r',encoding='utf-8') as f :
    reader =pd.read_csv(f,sep='|')  #原来的sep = ','
    date=reader.ix[:,2]
    descript=reader.ix[:,1]
    predicted=reader.ix[:,0]
    length=len(date)
    alldate=[]
    alldate.append(date[0])
    numdate=[]
    numdate.append(0)
    save=[]
    
    with open ('./中兴新闻矩阵_3.csv','a+',newline='', encoding = 'utf-8')as cf:
        writer=csv.writer(cf)
        writer.writerow(col)
        for i in range(1,length):
            if date[i]!=date[i-1]:
                alldate.append(date[i])
                numdate.append(i)

        for k in range(1,len(numdate)):
            count=[0]*len(labels)
            a=numdate[k-1]
            b=numdate[k]
            for i in range(a,b):
                for j in range(len(labels)):
                    if predicted[i]==labels[j]:
                        count[j]+=1
            save.append(count)
        for i in range(0,len(save)):
            total=[]
            total.append(alldate[i])
            for j in range(len(save[i])):
                total.append(save[i][j])
            writer.writerow(total)

        
   



           
        
    






  
            
            
        
   
    
            

    
