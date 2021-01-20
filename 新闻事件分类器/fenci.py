# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:24:08 2019

@author: wuzix
"""
import re
import jieba
import pandas as pd
import jieba.analyse
#from zhon.hanzi import punctuation

text1='D:\\毕业设计\\代码\\newtry\\新闻事件预测\\格力电器分词版.csv'
text2= 'D:\\毕业设计\\代码\\newtry\\新闻事件预测\\格力电器分词结果.csv'

jieba.load_userdict("D:\\毕业设计\\代码\\newtry\\stock_dict.txt")
jieba.analyse.set_stop_words('D:\\毕业设计\\代码\\newtry\\stopwords2.txt')

reg = "[^A-Za-z\u4e00-\u9fa5]"

f=pd.read_csv(text1, sep=',',encoding='utf-8',error_bad_lines=False)
f.columns = ['Descript']
outfile=open(text2,'w',encoding='utf-8')

def fenci(line):
    line = line.strip()
    line_result= re.sub(reg,"",line)
    return line_result

for line in f['Descript']:
    kk=fenci(line)
    seg_list = jieba.cut(kk)
    descript=''
    for word in seg_list:
        descript += word
        descript +=" "
    #descript = " ".join(seg_list)
    print(descript)
    outfile.write(descript+'\n')

outfile.close()


