import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np

import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

def train_cnn_rnn(input_file,training_config):
	epochs=10
#	input_file = sys.argv[1]
	x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file)

#	training_config = sys.argv[2]
	params = json.loads(open(training_config).read())

	# Assign a 300 dimension vector to each word
    #词嵌入
	word_embeddings = data_helper.load_embeddings(vocabulary)
    #给每一个词编号
	embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
	embedding_mat = np.array(embedding_mat, dtype = np.float32)

	# Split the original dataset into train set and test set
    #原来代码random_state=16
	x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1, random_state=0)

	# Split the train set into train set and dev set
	x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1, random_state=0)

	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

	# Create a directory, everything related to the training will be saved in this directory
	timestamp = str(int(time.time()))
	trained_dir = './训练结果/trained_results_' + timestamp + '/'
	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
	os.makedirs(trained_dir)

	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat=embedding_mat,
				sequence_length=x_train.shape[1],
				num_classes = y_train.shape[1],
				non_static=params['non_static'],
				hidden_unit=params['hidden_unit'],
				max_pool_size=params['max_pool_size'],
				filter_sizes=map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
			grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Checkpoint files will be saved in this directory during training
			checkpoint_dir = './训练结果/checkpoints_' + timestamp + '/'
			if os.path.exists(checkpoint_dir):
				shutil.rmtree(checkpoint_dir)
			os.makedirs(checkpoint_dir)
			checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def train_step(x_batch, y_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				_, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)

			def dev_step(x_batch, y_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				step, loss, accuracy, num_correct, predictions = sess.run(
					[global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
				return accuracy, loss, num_correct, predictions

			saver = tf.train.Saver(tf.all_variables())
			sess.run(tf.initialize_all_variables())

			# Training starts here
			train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			# Train the model with x_train and y_train
			for epoch in range(epochs):
				for train_batch in train_batches:
					x_train_batch, y_train_batch = zip(*train_batch)
					train_step(x_train_batch, y_train_batch)
					current_step = tf.train.global_step(sess, global_step)

					# Evaluate the model with x_dev and y_dev
					if current_step % params['evaluate_every'] == 0:
						dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

						total_dev_correct = 0
						for dev_batch in dev_batches:
							x_dev_batch, y_dev_batch = zip(*dev_batch)
							acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
							total_dev_correct += num_dev_correct
						accuracy = float(total_dev_correct) / len(y_dev)
						logging.info('Accuracy on dev set: {}'.format(accuracy))

						if accuracy >= best_accuracy:
							best_accuracy, best_at_step = accuracy, current_step
							path = saver.save(sess, checkpoint_prefix, global_step=current_step)
							logging.critical('Saved model {} at step {}'.format(path, best_at_step))
							logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
			logging.critical('Training is complete, testing the best model on x_test and y_test')

			# Save the model files to trained_dir. predict.py needs trained model files. 
            #二进制文件，存储了weights，biases，gradients等变量
			saver.save(sess, trained_dir + "best_model.ckpt")

			# Evaluate x_test and y_test
			saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
			acc, loss, num_test_correct, predictions = dev_step(x_test, y_test)
			from sklearn.metrics import recall_score
			from sklearn.metrics import f1_score
			from sklearn.metrics import accuracy_score
     
			y_test=[np.argmax(y_t) for y_t in y_test]      
			#print(sorted(list(set(y_test))))       
				   
			recall_l=recall_score(y_test,predictions,average=None)
			f1_score=f1_score(y_test,predictions,average=None)
			acc_score=accuracy_score(y_test,predictions)
			total_test_correct = int(num_test_correct)                
			logging.critical('Recall on test set: '+str(recall_l))
			logging.critical('Acc on test set: '+str(acc_score))
			logging.critical('F1 on test set: '+str(f1_score))
			logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))
			print(len(labels))
			print(len(recall_l))
			print(len(f1_score))
			labels_=[labels[n] for n in sorted(list(set(y_test)))]
				
			logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))
			df_=pd.DataFrame();df_["labels"]=labels_;df_["recall"]=recall_l;df_["f1"]=f1_score;df_.to_csv(trained_dir+'matrics.csv',index=False)
			# Save trained parameters and files since predict.py needs them
			#print (vocabulary)

	with open(trained_dir + 'words_index.json', 'w') as outfile:
		#jsObj = json.dumps(vocabulary)  
		#outfile.write(jsObj)  
		#outfile.close()  
        #将dict类型的数据转成str，并写入到json文件,indent表示缩进4，ensure_ascii表示输出真正的中文
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
        #pickle.dump()序列化对象，将对象obj保存到文件file中去
		pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
	with open(trained_dir + 'labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)

	params['sequence_length'] = x_train.shape[1]
	with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
	# python3 train.py ./data/train.csv.zip ./training_config.json
	train_cnn_rnn("./data_sample.zip","./training_config.json")
