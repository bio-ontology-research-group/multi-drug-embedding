
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import pandas as pd
import itertools
import pdb
import json

import tensorflow as tf
import random


np.random.seed(33)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
random.seed(12345)



def negcum(rank_vec):
	rank_vec_cum = []
	prev = 0
	for x in rank_vec:
		if x == 0:
			x = x+1
			prev = prev + x
			rank_vec_cum.append(prev)
		else:
			rank_vec_cum.append(prev)
	rank_vec_cum = np.array(rank_vec_cum)
	return rank_vec_cum




data_dir = 'data/'

with open(data_dir+'drugs2ind_doid.dict','r') as f:
	drugs_targets = json.load(f)


data = pd.read_csv(data_dir+'drugs_embeddings_combined_has_indication.txt', header = None, sep = ' ')
embds_data = data.values
drugs_dict = dict(zip(embds_data[:,0],embds_data[:,1:]))


data = pd.read_csv(data_dir+'diseases_embeddings_combined_has_indication.txt', header = None, sep = ' ')
embds_data = data.values
tar_dict = dict(zip(embds_data[:,0],embds_data[:,1:]))

drugs_embeddings = {}
targets_embeddings = {}


for item in drugs_dict:	
	ent = item.split('/')[-1].strip('>')
	if ent.startswith('CID'):
		drugs_embeddings[ent] = np.array(drugs_dict[item], dtype = 'float32')

for item in tar_dict:
	ent = item.split('/')[-1].strip('>')
	if ent.startswith('DOID_'):
		targets_embeddings[ent] = np.array(tar_dict[item], dtype = 'float32')


positive_drug_tar = {}

common_drugs = np.genfromtxt('common_drugs_indications.txt', dtype = 'str')
common_genes = np.genfromtxt('common_diseases_indications.txt', dtype = 'str')


for drug in drugs_targets:
	tars = drugs_targets[drug]
	for tar in tars:
		if drug in drugs_embeddings and drug in common_drugs and tar in targets_embeddings and tar in common_genes:		
			drug_embds = drugs_embeddings[drug]
			tar_embds = targets_embeddings[tar]
			positive_drug_tar[(drug,tar)] = np.concatenate((drug_embds,tar_embds), axis=0)


allpairs = list(itertools.product(set(common_drugs),set(common_genes)))

positive_pairs = positive_drug_tar.keys()
negative_pairs = set(allpairs) - set(positive_pairs)
negative_pairs = random.sample(negative_pairs, len(positive_pairs))

negative_drug_tar = {}


for (drug,tar) in negative_pairs:
	if drug in drugs_embeddings and tar in targets_embeddings:
		drug_embds = drugs_embeddings[drug]
		tar_embds = targets_embeddings[tar]
		negative_drug_tar[(drug,tar)] = np.concatenate((drug_embds,tar_embds), axis=0)


train_pairs = np.genfromtxt('common_indications_pair_train.txt', dtype = 'str')
test_pairs = np.genfromtxt('common_indications_pair_test.txt', dtype = 'str')

# pdb.set_trace()

train_data = []
train_labels = []
test_data = []
test_labels = []
train_pos = set()
test_pos = set()

for item in train_pairs:
	pair = (item[0],item[1])
	if pair in positive_drug_tar:
		train_data.append(positive_drug_tar[pair])
		train_labels.append(1)
		train_pos.add(item[0])
	elif pair in negative_drug_tar:
		train_data.append(negative_drug_tar[pair])
		train_labels.append(0)


for item in test_pairs:
	pair = (item[0],item[1])
	if pair in positive_drug_tar:
		test_data.append(positive_drug_tar[pair])
		test_labels.append(1)
		test_pos.add(item[0])
	elif pair in negative_drug_tar:
		test_data.append(negative_drug_tar[pair])
		test_labels.append(0)

train_data = np.array(train_data)
test_data = np.array(test_data)
# pdb.set_trace()
tr_labels = keras.utils.to_categorical(train_labels, num_classes=None)
ts_labels = keras.utils.to_categorical(test_labels, num_classes=None)


tf.set_random_seed(33)
model = Sequential()
model.add(Dense(512, input_dim=256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model.fit(train_data, tr_labels,validation_split = 0.1,epochs=100,callbacks=[earlystopper])

pred = model.predict_proba(test_data)[:,1]

print 'AUC: {}'.format(roc_auc_score(test_labels, pred))

fpr, tpr, thresholds = roc_curve(test_labels, pred)
auc_data2 = np.c_[fpr, tpr]

label_mat = {}
recall_100 = {}
recall_10 = {}
ranked_tars10 = {}
ranked_tars100 = {}
allranked = {}

ff = open('indications_ranked_multimodalII.txt','w')

for drug in test_pos:
	tars = drugs_targets[drug]
	s2 = set(common_genes)
	if drug in train_pos: #exclude training associations
		train_tar = [item[1] for item in train_pairs if item[0]== drug]
		test_tar = list(set(tars) - set(train_tar))

		s1 = list(set(test_tar))
	else:
		s1 = list(set(tars))

	if set(s1).intersection(s2):
		drug_embed = list()
		if drug in drugs_embeddings:
			drug_embed.append(drugs_embeddings[drug])
			drugembds = np.array(drug_embed, dtype='float32')

			test_emds = []
			to_test = list(set(common_genes) - set(train_tar))
			for tar in to_test:
				taremds = targets_embeddings[tar]
				pair_embds = np.concatenate((drugembds[0], taremds), axis=0)
				test_emds.append(pair_embds)

			test_emds = np.array(test_emds, dtype='float32')
			y_pred = model.predict_proba(test_emds)[:,1]
			sorted_idx = np.argsort(y_pred)[::-1]
			sort_tars = [to_test[arg] for arg in sorted_idx]
			
			label_vec = [0]*len(sort_tars)
			test_ranks = []
			ranked_tars = []

			ff.write(drug+':')
			for ind in s1:
				if ind in sort_tars:
					idx = sort_tars.index(ind)
					label_vec[idx] = 1
					test_ranks.append(idx)
					ranked_tars.append(ind)
					ff.write(str(ind)+' ranked at '+str(idx)+', ')
			ff.write('\n')

			label_mat[drug] = label_vec

			test_r = np.array(test_ranks,dtype='int32')
			if len(test_ranks) > 0:
				ranked_tars = np.array(ranked_tars)
				recall_100[drug] = len(np.where(test_r <= 100)[0])/float(len(test_r))
				recall_10[drug] = len(np.where(test_r <= 10)[0])/float(len(test_r))
				ranked_tars10[drug] = ranked_tars[np.where(test_r <= 10)[0]].tolist()
				ranked_tars100[drug] = ranked_tars[np.where(test_r <= 10)[0]].tolist()
				allranked[drug] = ranked_tars.tolist()

ff.close()

#get max label vec dimension to compare with
len_vec = []
for item in label_mat:
	len_vec.append(len(label_mat[item]))


col_num = max(len_vec)
array_tp = np.zeros((len(label_mat), col_num),dtype='float32')
array_fp = np.zeros((len(label_mat), col_num), dtype = 'float32')

for i,row in enumerate(label_mat.values()):
		elem = np.asarray(row, dtype='float32')
		tofill = col_num - len(row)
		tpcum = np.cumsum(elem)
		tpcum = np.append(tpcum,np.ones(tofill)*tpcum[-1])
		fpcum = negcum(elem)
		fpcum = np.append(fpcum,np.ones(tofill)*fpcum[-1])
		array_tp[i] = tpcum
		array_fp[i] = fpcum


#compute fpr and tpr Rob's way 
tpsum = np.sum(array_tp, axis = 0)
fpsum = np.sum(array_fp, axis = 0)
tpr_r = tpsum/max(tpsum)
fpr_r = fpsum/max(fpsum)

auc_data2 = np.c_[fpr_r, tpr_r]

print('Number of drugs: {}'.format(len(label_mat)))
print('Number of targets: {}'.format(len(common_genes)))
print('auc:  {}'.format(auc(fpr_r, tpr_r)))
print('average recall@100:{}'.format(np.mean(recall_100.values())))
print('average recall@10:{}'.format(np.mean(recall_10.values())))	
pdb.set_trace()
