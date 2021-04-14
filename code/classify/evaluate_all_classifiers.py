import re
import os
import numpy as np
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyClassifier
from collections import defaultdict
from sklearn.metrics import classification_report
from ast import literal_eval
from scipy.stats import pearsonr



def load_data(train_path,eval_path,frame_type):
	all_data = defaultdict(list)
	train_file = os.path.join(train_path,frame_type + '.tsv')
	df_train = pd.read_csv(train_file,sep='\t',converters={'labels': literal_eval})

	eval_file = os.path.join(eval_path,frame_type + '.tsv')
	df_eval = pd.read_csv(eval_file,sep='\t',converters={'labels': literal_eval})

	all_data['X_train'] = df_train.text
	all_data['X_eval'] = df_eval.text

	all_data['y_train_list'] = df_train['labels']
	all_data['y_eval_list'] = df_eval['labels']

	all_data['y_train'] = df_train.drop(columns=['Unnamed: 0','text','id','labels'])
	all_data['y_eval'] = df_eval.drop(columns=['Unnamed: 0','text','id','labels'])
	all_data['labels'] = list(all_data['y_eval'].columns)

	return all_data

def build_or_load_model(model_name,frame_type,seed):
	if model_name == 'dummy_random':
		clf = DummyClassifier(strategy='uniform',random_state=seed)
	if model_name == 'dummy_stratified':
		clf = DummyClassifier(strategy='stratified',random_state=seed)
	if model_name == 'dummy_frequent':
		clf = DummyClassifier(strategy='most_frequent',random_state=seed)
	if model_name == 'logreg_unigram':
		clf = MultiOutputClassifier(LogisticRegression(solver='saga',random_state=seed))
	if model_name == 'logreg_bigram':
		clf = MultiOutputClassifier(LogisticRegression(solver='saga',random_state=seed))

	model_path_base = '/shared/2/projects/framing/models/classify/'
	model_args = {"reprocess_input_data": True,'no_cache': True}
	if model_name == 'roberta_finetune':
		model_path = os.path.join(model_path_base,frame_type,f'11-03-20_60_epochs_default_thresh_{seed}_seed')
		clf = MultiLabelClassificationModel('roberta',model_path,cuda_device=1,args=model_args)
	if model_name == 'roberta_baseline':
		model_path = os.path.join(model_path_base,frame_type,f'roberta_baseline_11-05-20_60_epochs_default_thresh_{seed}_seed')
		clf = MultiLabelClassificationModel('roberta',model_path,cuda_device=1,args=model_args) 
	# if model_name == 'roberta_all':
	# 	model_path = os.path.join(model_path_base,'all_frames',f'10-08-20_60_epochs_default_thresh_{seed}_seed')
	# 	clf = MultiLabelClassificationModel('roberta',model_path,cuda_device=0) 

	return clf


def run_model(model_name,clf,data):
	X_train = data['X_train']
	X_eval = data['X_eval']

	if model_name.startswith('logreg') or model_name.startswith('dummy'):
		if model_name == 'logreg_unigram':
			vec = CountVectorizer(ngram_range=(1, 1), lowercase=True)
			X_train = vec.fit_transform(data['X_train'])
			X_eval = vec.transform(data['X_eval'])
		elif model_name == 'logreg_bigram':
			vec = CountVectorizer(ngram_range=(1, 2), lowercase=True)
			X_train = vec.fit_transform(data['X_train'])
			X_eval = vec.transform(data['X_eval'])			
		clf.fit(X_train,data['y_train'])
		
	y_pred = clf.predict(X_eval)
	#y_pred = np.array([np.array(s) for s in y_pred]).astype(int)

	y_true = data['y_eval']
	if model_name.startswith('roberta'):
		y_pred = pd.DataFrame((y_pred[0]))
		y_pred.columns=y_true.columns
		


	#full_y_true = np.array(eval_df[labels].astype(int))
	#full_y_pred = np.array(predictions)

	return y_true, y_pred
	

def evaluate_model(y_true,y_pred,data):
	report = classification_report(y_true,y_pred,target_names=data['labels'],output_dict=True)

	all_f1 = [report[f]['f1-score'] for f in sorted(data['labels'])]
	all_support = [report[f]['support'] for f in sorted(data['labels'])]
	f1_support_corr = pearsonr(all_f1,all_support)
	lrap = label_ranking_average_precision_score(y_true,y_pred)
	

	df = pd.DataFrame(report).transpose()
	df['LRAP'] = [lrap] * len(df)
	df['Support Pearson'] = [f1_support_corr] * len(df)



	correct_by_sample = pd.DataFrame((y_true==y_pred))
	return df,correct_by_sample




def main():
	frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative']
	model_names = ['dummy_random','dummy_stratified','dummy_frequent',
					'logreg_unigram','logreg_bigram',
					'roberta_finetune','roberta_baseline']

	seeds = [35, 12, 45, 42, 23]

	base_path = '/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta'
	train_path = os.path.join(base_path,'train')

	#eval_subsets = ['dev','test']
	eval_subsets = ['eval_liberal','eval_conservative','eval_US','eval_GB','eval_EU',]

	for eval_set in eval_subsets:
		eval_path = os.path.join(base_path,eval_set)
		base_out_path = f'/shared/2/projects/framing/models/classify/3-30-21_{eval_set}/'
		if not os.path.exists(base_out_path):
			os.mkdir(base_out_path)
		for frame_type in frame_types:
			for model_name in model_names:
				for seed in seeds:
					print(eval_set,frame_type,model_name,seed)
					data = load_data(train_path,eval_path,frame_type)
					model = build_or_load_model(model_name,frame_type,seed=seed)
					y_true,y_pred = run_model(model_name,model,data)
					df,correct_by_sample = evaluate_model(y_true,y_pred,data)
					frame_type_path = os.path.join(base_out_path,frame_type)
					if not os.path.exists(frame_type_path):
						os.mkdir(frame_type_path)
					out_path = os.path.join(frame_type_path,model_name)
					if not os.path.exists(out_path):
						os.mkdir(out_path)
					out_file = os.path.join(out_path,f'seed_{seed}.tsv')
					out_file_by_sample = os.path.join(out_path,f'seed_{seed}_by_sample.tsv')
					df.to_csv(out_file,sep='\t')
					correct_by_sample.to_csv(out_file_by_sample,sep='\t')

if __name__ == "__main__":
	main()