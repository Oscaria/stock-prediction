# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:40:02 2020

@author: AAA
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:18:03 2020

@author: xuyuemei
"""


import pandas as pd 
import csv
import numpy as np
#from sklearn import svm,metrics,cross_validation 改 xu
from sklearn import svm,metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import matplotlib.pylab as plt
from sklearn.utils.validation import column_or_1d
from sklearn.ensemble import GradientBoostingClassifier
import os

'''
gbdt_test_acc=[]
gbdt_test_rec=[]
gbdt_test_f1=[]
gbdt_train_acc=[]
gbdt_train_rec=[]
gbdt_train_f1=[]
'''

'''
file=open('./Zhongxing财务_删0_label0.01_zscore_afterMerge_event+senti_-1.csv',encoding='utf-8')
#file=open('./GeliFinance_no0_label0.01_zscore_afterMerge_event+senti_-1.csv',encoding='utf-8')

data=pd.read_csv(file)
x=data.iloc[:,1:96] #通过修改这个，改变输入
y=data[['label']]
Data=x.values.tolist()
value=y.values.tolist()
'''

#data_dir = '../建立财务矩阵/threshold/afterMerge/event+senti_-1'
data_dir = '../../../../格力_再处理/代码/个股走势预测/建立财务矩阵/threshold/afterMerge/event+senti_-1'
ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]

for th in ths:
    print(th)
    
    #file=open('./Zhongxing财务_删0_label0.01_zscore_afterMerge_event+senti_-1.csv',encoding='utf-8')
    #file=open('./GeliFinance_no0_label0.01_zscore_afterMerge_event+senti_-1.csv',encoding='utf-8')
    
    #file = os.path.join(data_dir,'Zhongxing财务_删0_label'+str(th)+'_zscore_afterMerge_event+senti.csv')
    file = os.path.join(data_dir,'GeliFinance_no0_label'+str(th)+'_zscore_afterMerge_event.csv')
    
    data=pd.read_csv(file)
    x=data.iloc[:,1:96] #通过修改这个，改变输入
    y=data[['label']]
    Data=x.values.tolist()
    value=y.values.tolist()
    
    L=len(x)  # 训练集的长度
    
    gbdt_test_acc=[]
    gbdt_test_rec=[]
    gbdt_test_f1=[]
    gbdt_train_acc=[]
    gbdt_train_rec=[]
    gbdt_train_f1=[]
    
    train =set_train= 13 #How many data for train, 9 is the least.
    
    #Data['Value']=value
    correct = 0
    train_original=train
    i=0
    sample=[]
    target=[]
    predict=[]
    real=[]
    #print('@@@', Data[3:5])
    while train<L-set_train:
        Data_train=Data[train-train_original:train]
        #print("##########", train)
        sample.append(Data_train)
        value_train=value[train]
        #print('@@@', value_train)
        target.append(value_train)
        train = train+set_train
    
    #print(len(target))
    #print(len(sample))
    #print(train_original)
    target=np.array(target) # 训练的label 
    sample=np.array(sample).reshape(target.shape[0],-1)  # 训练数据
    
    
    
    
    for i in range(0,100):
        train_data,test_data,train_label,test_label = train_test_split(sample,target,random_state=i,train_size=0.8,test_size=0.2)
     
        model= GradientBoostingClassifier()
        model.fit(train_data,train_label.ravel())
            
        #grid = joblib.load('grid3.pkl')
        pre_train=model.predict(train_data)
        pre_test=model.predict(test_data)
        
        '''
        print ("第",i,"round：")
        print("训练集acc：", accuracy_score(train_label,pre_train))
        print("测试集acc：", accuracy_score(test_label,pre_test))
        print("训练集recall：", recall_score(train_label,pre_train))
        print("测试集recall：", recall_score(test_label,pre_test))
        print("训练集f1：", metrics.f1_score(train_label,pre_train))
        print("测试集f1：", metrics.f1_score(test_label,pre_test))
        print("---------分割线-----------------")
        '''
        gbdt_train_acc.append(accuracy_score(train_label,pre_train))
        gbdt_test_acc.append(accuracy_score(test_label,pre_test))
        gbdt_train_rec.append(recall_score(train_label,pre_train))
        gbdt_test_rec.append(recall_score(test_label,pre_test))
        gbdt_train_f1.append(metrics.f1_score(train_label,pre_train))
        gbdt_test_f1.append(metrics.f1_score(test_label,pre_test))
    
    # 求平均值
    def avgcal(listarr):
        num=len(listarr)
        sumvalue=0
        for i in listarr:
            sumvalue+=i
        return sumvalue/num
    print("---------分割线-----------------")
    print('@@@', th)
    print("平均训练集acc：", avgcal(gbdt_train_acc))
    print("平均训练集recall：", avgcal(gbdt_train_rec))
    print("平均训练集f1：", avgcal(gbdt_train_f1))
    print("---------分割线-----------------")
    print("平均测试集acc：", avgcal(gbdt_test_acc))
    print("平均测试集recall：", avgcal(gbdt_train_rec))
    print("平均测试集f1：", avgcal(gbdt_train_f1))
    print(len(gbdt_test_acc))
    
    
    #print(gbdt_test_acc)
        