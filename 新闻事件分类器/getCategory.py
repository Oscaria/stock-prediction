# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:03:14 2017

@author: think
"""
import jieba 
import re
import pandas as pd
import csv


import os
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN

logging.getLogger().setLevel(logging.INFO)

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep='|')
	select = ['Descript']

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: x.split(' ')).tolist()

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'Category' in df.columns:
		select.append('Category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()
	if 'Date' in df.columns:
		select.append('Date')
	if 'Bid' in df.columns:
		select.append('Bid')
	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df

def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

def predict_unseen_data(trained_dir,test_file,date):
	#trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	#test_file = sys.argv[2]

	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
	x_, y_, df = load_test_data(test_file, labels)
	x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index)

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './预测结果/predicted_results_' + timestamp + '/'

	if not os.path.exists(predicted_dir):
		os.makedirs(predicted_dir)
	

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],
				sequence_length = len(x_test[0]),
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = len(labels),
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def predict_step(x_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			logging.critical('{} has been loaded'.format(checkpoint_file))

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			df['PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=True)
			df.to_csv(predicted_dir + 'predictions_all_'+date+'.csv', index=False, columns=columns, sep='|')
			#logging.info("Here is the result : {}".format(df))
        
			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
				logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

if __name__ == '__main__':
	# python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv
    #载入自定义词典
    temp_file="./temp_file.csv"
    jieba.load_userdict("./stock_dict.txt")
    stop_words=pd.read_csv('./stopwords2.txt', lineterminator="\n", encoding='utf-8',header=None,error_bad_lines=False)
    stopwords=(stop_words[0]).tolist()
    title_file='../../数据集/中兴通讯新闻分词.csv'
    #title_file='D:\\毕业设计\\代码\\StockForecast-master\\cnn-rnn\\all.csv'
    train_dir="./训练结果/trained_results_1594750510/"
    date="2020-07-15-3"
    df=pd.read_csv(title_file, sep=',',encoding="utf-8",engine="python")
    
    outs=[]
    outs.append(list(df.columns))
    for i in range(len(df)):
        out=""
        #print (text)
        text=df.iloc[i,2]
        text2=text
        
        if "\'" in text2:
            #print(text2)
            idx_1=text2.index("\'")
            #print(idx_1)
            text2=text2[idx_1+1:]
            #print(text2)
            idx_2=text2.index("\'")
            out+=text2[:idx_2]+" "
            text=text[:idx_1]+text[idx_2+1:]
        
        text=re.sub("[\s+\.\!\/_,:–?$%^(+)*\'\"\d]+|[㎡{}\|§．×÷Οαβχ“”’‘ⅠⅡⅢⅣⅧ∶≤≥⑴⑵⑶○◎〇〈〉「」《》<>\[\]■□％⊙;〔〕①②③④⑤⑥⑦⑧⑨⑩√▼●/←→+——·！，。？、~@#￥%……&（）：；【】>=-]+","",text)
        reg = "[^A-Za-z\u4e00-\u9fa5]"
        text=re.sub(reg,"",text)
        text=re.sub('月日',"",text)
        text= re.sub('A股',"",text)
        text= re.sub('亿元',"",text)
        seg_list=jieba.cut(text,cut_all=False)
        #停用词过滤
        for word in seg_list:
            if len(word)>1:
                if word not in stopwords:
                    out+=word+" "
        
        out+=" ".join(seg_list)
        outs.append((df.iloc[i,0],df.iloc[i,1],out))
    with open(temp_file,"w",newline="",encoding="utf8") as csvfile:
        writer=csv.writer(csvfile,delimiter='|')
        writer.writerows(outs)
    #df=pd.DataFrame(out)
    #pd.DataFrame(out).to_csv(header=False,index=False)

    predict_unseen_data(train_dir,temp_file,date)