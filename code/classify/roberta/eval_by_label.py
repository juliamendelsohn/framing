from simpletransformers.classification import MultiLabelClassificationModel
import os
from ast import literal_eval
import pandas as pd
from sklearn.metrics import classification_report,f1_score,label_ranking_average_precision_score
import torch
import argparse
import numpy as np

device = 1

all_scores = []
category_scores = []
model_path_base = f'/shared/2/projects/framing/models/classify/'
eval_path = f'/shared/2/projects/framing/data/labeled_data/splits_08-16-20/roberta/dev/'

frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
for frame_type in frame_types:
	all_path = os.path.join(model_path_base,'all_frames','09-24-20_60_epochs_default_thresh')
	model_path = os.path.join(model_path_base,frame_type,'09-24-20_60_epochs_default_thresh')
	model = MultiLabelClassificationModel('roberta',model_path,cuda_device=device) 
	eval_file = os.path.join(eval_path,frame_type+'.tsv')
	eval_df = pd.read_csv(eval_file,sep='\t',converters={'labels': literal_eval})
	labels = list(eval_df.columns)[3:-1]
	predictions, raw_outputs = model.predict(list(eval_df['text']))


	full_y_true = np.array(eval_df[labels].astype(int))
	full_y_pred = np.array(predictions)


	print(full_y_true.shape)
	print(full_y_pred.shape)

	cat_f1 = f1_score(full_y_true,full_y_pred,average='macro') 
	cat_lrap = label_ranking_average_precision_score(full_y_true,full_y_pred)
	category_scores.append((frame_type,cat_f1,cat_lrap))


	for i,label in enumerate(labels):
		y_true = np.array(eval_df[label].astype(int))
		y_pred = np.array([predictions[j][i] for j in range(len(predictions))])
		score = f1_score(y_true,y_pred,average='macro')
		print(frame_type,label,score)
		all_scores.append((frame_type,label,score))


df_cat = pd.DataFrame(category_scores,columns = ['Frame Type','Macro F1','LRAP'])
df_cat.to_csv(f'/shared/2/projects/framing/models/classify/09-24-20_overall_eval.tsv',sep='\t')


df = pd.DataFrame(all_scores, columns = ['Frame Type','Frame', 'Macro F1'])
df.to_csv(f'/shared/2/projects/framing/models/classify/09-24-20_eval_by_class_macrof1.tsv',sep='\t')


	