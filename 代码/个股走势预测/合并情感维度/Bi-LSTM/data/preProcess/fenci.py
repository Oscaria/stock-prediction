# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:24:08 2019

@author: xuyuemei
"""
import re
import jieba
import pandas as pd
import jieba.analyse
#from zhon.hanzi import punctuation



text1='./TechNews_modified_500.csv'
text2= '../data/preProcess/TechNews_modified_500_clean.csv'

#jieba.load_userdict("../data/preProcess/stock_dict.txt")
jieba.analyse.set_stop_words('./stopwords.txt')

reg = "[^A-Za-z\u4e00-\u9fa5]"

f=pd.read_csv(text1, sep=',',encoding='utf-8',error_bad_lines=False)
title=f.iloc[:,2]
#f.columns = ['Title']
#outfile=open(text2,'w',encoding='utf-8')

results=pd.DataFrame()
news=[]

def fenci(line):
    print (line)
    line = line.strip()
    line_result= re.sub(reg,'',line)
    return line_result

for line in title:
    kk=fenci(line)
    seg_list = jieba.cut(kk)
    descript=''
    for word in seg_list:
        descript += word
        descript +=" "
    #descript = " ".join(seg_list)
    print(descript)
    
    news.append(descript)
    
    #outfile.write(descript+'\n')
results['review']=news
results['rate']=f.iloc[:,3]
results.to_csv('./TechNews_modified_500_fenci.csv',index=False,encoding='utf-8')
#outfile.close()

testdata=pd.DataFrame()
testdata=results[0:1000]
testdata.to_csv('./xdataChina_modified.csv',index=False,encoding='utf-8')
'''
x=[]
for i in range(1,101):
    x.append(results[i,0])
testdata['review']=x'''
#testdata.to_csv('../data/preProcess/xdataChina.csv',index=False,encoding='utf-8')

#df = pd.read_csv('../data/preprocess/xdataChina.csv', encoding = 'utf-8',error_bad_lines=False) 
#print (df.iloc[1,0])
