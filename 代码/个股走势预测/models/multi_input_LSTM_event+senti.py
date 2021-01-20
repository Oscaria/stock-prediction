# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:11:57 2020

@author: wuzix
"""


import pandas as pd
import numpy as np
import keras
from keras import models
from keras import layers
from keras import Input
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
import os

def generator(data, lookback, step, min_index, max_index, batch_size):
        i=min_index+lookback
        while 1:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i, min(i+batch_size, max_index), step)
            np.random.shuffle(rows)
            i += batch_size
            
            samples1=np.zeros((len(rows), lookback, 12))
            samples2=np.zeros((len(rows), lookback, 83))
            targets=np.zeros((len(rows), ))
            for j, row in enumerate(rows):
                indices=range(row-lookback, row)
                samples1[j]=data[indices,:12]
                samples2[j]=data[indices,12:data.shape[1]-1]
                targets[j]=data[row][data.shape[1]-1]
            yield {'num':samples1,'text':samples2}, targets


model_path='./model/zx_event+senti_0.01_删0.h5'
data_dir = '../建立财务矩阵/threshold/afterMerge/event+senti_-1'

ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
#ths=[0.01]

for th in ths:
    print(th)
    fname =os.path.join(data_dir,'Zhongxing财务_删0_label'+str(th)+'_zscore_afterMerge_event+senti.csv')
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
    
    #lookback=12
    #step=14
    #batch_size=140
    
    lookback=13  
    step=14 
    batch_size=140
    bound=bound=train_size*3//4
    callbacks_list=[
               keras.callbacks.EarlyStopping(monitor='acc', patience=1,),
               keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                             monitor='val_loss', save_best_only=True,)]
    
    
    train_gen=generator(train_data, lookback=lookback, step=step, min_index=0, max_index=bound, batch_size=batch_size)
    val_gen=generator(train_data, lookback=lookback, step=step, min_index=bound+1, max_index=len(train_data), batch_size=batch_size)
    test_gen=generator(test_data, lookback=lookback, step=1, min_index=0, max_index=len(test_data), batch_size=98)
    #all_gen=generator(all_data, lookback=lookback, step=step, min_index=0, max_index=len(all_data), batch_size=batch_size)
            
    train_steps=(bound-lookback)//batch_size
    val_steps=(len(train_data)-bound+1-lookback)//batch_size
    test_steps=(len(test_data)-lookback)//98
    #all_steps=(len(all_data)-lookback)//batch_size
    
    num_input=Input(shape=(None,12),dtype='float32',name='num')
    encoded_num=layers.LSTM(64)(num_input)
    text_input=Input(shape=(None,83),dtype='float32',name='text')
    encoded_text=layers.LSTM(128)(text_input)
    concatenated=layers.concatenate([encoded_num,encoded_text],axis=-1)
    concatenated=layers.Dense(128,activation='relu')(concatenated)
    
    
    
    preds=layers.Dense(1,activation='sigmoid')(concatenated)
    model=models.Model([num_input,text_input],preds)
    model.compile(optimizer=RMSprop(),metrics=['acc'],loss='binary_crossentropy')
    history=model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=20, 
                                    validation_data=val_gen, validation_steps=val_steps,
                                    callbacks=callbacks_list, verbose=0)
    
    model=models.load_model(model_path)
    results1=model.evaluate_generator(train_gen, steps=train_steps)
    results2=model.evaluate_generator(val_gen, steps=val_steps)
    results3=model.evaluate_generator(test_gen,steps=test_steps)
    print (model.metrics_names)
    print('event+senti_-1', th, lookback)
    print(results1)
    print(results2)
    print(results3)
    
    preds2=model.predict_generator(test_gen, steps=test_steps)[:,0]
    targets2=[]
    
    '''
    preds2=model.predict_generator(test_gen, steps=test_steps)[:,0]
    targets2=[]
    dates2=[]
    close2=[]
    close3=[]
    
    test_data2=data.iloc[train_size:,0:].fillna(0)
    temp=pd.read_csv('../建立财务矩阵/threshold/afterMerge/Zhongxing财务_删0_label0.01_forVisual_afterMerge.csv',encoding='utf-8')
    
    close=temp.iloc[train_size:,5].fillna(0)
    close=np.array(close,dtype=float)
    close=np.asarray(close).astype('float32')
    
    for i in range(1,test_steps*98+1):
        targets2.append(test_data[i,test_data.shape[1]-1])
        close2.append(test_data[i,3])
        close3.append(close[i])
        dates2.append(test_data2.iloc[i,0])
    targets2=np.array(targets2)
    dates2=np.array(dates2)
    results2=pd.DataFrame()
    print(len(targets2))
    print(len(preds2))
    print(len(dates2))
    print('@@@',train_data[1,3] )
    results2["preds"]=preds2
    results2["targets"]=targets2
    results2['dates']=dates2
    results2['close']=close3
    results2.to_csv('./test/Zhongxing_event+senti_'+str(th)+'.csv',encoding='utf-8', index=False)
    '''