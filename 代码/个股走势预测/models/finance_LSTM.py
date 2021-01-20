
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:09:31 2020

@author: wuzix
"""
import os
import pandas as pd
import numpy as np
import keras
from keras import models
from keras import layers
#import keras.utils.Sequence
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


def generator(data, lookback, step, min_index, max_index, batch_size):
    i=min_index+lookback
    while 1:
        if i+batch_size>=max_index:
            i=min_index+lookback
        rows=np.arange(i, min(i+batch_size, max_index), step) #把每一个batch里面的数据预测数据找出来
        i += batch_size
            
        samples=np.zeros((len(rows), lookback, 12))
        targets=np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices=range(rows[j]-lookback, rows[j])
            samples[j]=data[indices,:12]   #其中 samples 是输入数据的一个批量，targets 是对应的目标温度数组
            targets[j]=data[rows[j]][train_data.shape[1]-1]
        yield samples, targets


model_path='./model/zx_finance_0.01_删0.h5'
data_dir = '../建立财务矩阵/threshold/afterMerge'

ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]

for th in ths:
    print(th)
    fname =os.path.join(data_dir,'Zhongxing财务_删0_label'+str(th)+'_zscore_afterMerge.csv')
    data=pd.read_csv(fname,encoding='utf-8')
    num=len(data)-1 #the first column is title
    train_size=1297
    #test_size=num-train_size
    # 测试数据的长度
    train_data=data.iloc[1:train_size,1:].fillna(0)
    train_data=np.array(train_data,dtype=float)
    train_data=np.asarray(train_data).astype('float32')
    
    test_data=data.iloc[train_size:,1:].fillna(0)
    test_data=np.array(test_data,dtype=float)
    test_data=np.asarray(test_data).astype('float32')
    
    
    lookback=12 # 根据过去13天的数据，预测第14天的股价
    step=14  # 数据采样的周期
    batch_size=140 # 每批量的样本数
    bound=bound=len(train_data)*3//4 # 训练数据占总数量的多少
    
    
    callbacks_list=[
               keras.callbacks.EarlyStopping(monitor='acc', patience=1,),
               keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                             monitor='val_loss', save_best_only=True,)]
    
    
    
    train_gen=generator(train_data, lookback=lookback, step=step, min_index=0, max_index=bound, batch_size=batch_size)
    val_gen=generator(train_data, lookback=lookback, step=step, min_index=bound+1, max_index=len(train_data), batch_size=batch_size)
    test_gen=generator(test_data, lookback=1, step=step,min_index=0, max_index=len(test_data), batch_size=batch_size)
    all_gen=generator(train_data, lookback=lookback, step=step, min_index=0, max_index=len(train_data), batch_size=batch_size)
     
    train_steps=(bound-lookback)//batch_size
    val_steps=(len(train_data)-bound+1-lookback)//batch_size
    test_steps=(len(test_data)-1)//batch_size
    
    all_steps=(len(train_data)-lookback)//batch_size
        
    # dropout LSTM
    
    model=models.Sequential()
    #model.add(layers.LSTM(64, input_shape=(None, 12)))
    model.add(layers.LSTM(64,input_shape=(None, 12)))
    model.add(layers.Dense(1,activation='sigmoid'))
    #model.compile(optimizer=RMSprop(),metrics=['acc'],loss='mae')
    model.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['acc'])
    history=model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=20, 
                                  validation_data=val_gen, validation_steps=val_steps,
                                   callbacks=callbacks_list, verbose=0)
    
    model=models.load_model(model_path)
    results1=model.evaluate_generator(train_gen, steps=train_steps)
    results2=model.evaluate_generator(val_gen, steps=val_steps)
    results3=model.evaluate_generator(test_gen,steps=test_steps)
    print (model.metrics_names)
    print('finance', th, lookback)
    print(results1)
    print(results2)
    print(results3)
