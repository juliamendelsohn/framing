from simpletransformers.classification import MultiLabelClassificationModel
import os
from ast import literal_eval
import pandas as pd
import sklearn
import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def load_data(train_file,eval_file):
	train_df = pd.read_csv(train_file,sep='\t',converters={'labels': literal_eval})
	eval_df = pd.read_csv(eval_file,sep='\t',converters={'labels': literal_eval})
	return train_df,eval_df


def configure_model_args(num_epochs,num_labels,threshold_setting,output_dir,manual_seed):
	if threshold_setting == 'default':
		threshold = 0.5
	elif threshold_setting == 'lower':
		threshold = 1.0 / num_labels
	hyperparams = {
		'num_train_epochs':num_epochs,
		'evaluate_during_training':True,
		'fp16': False,
		"use_early_stopping": True,
		"early_stopping_delta": 0,
		"early_stopping_patience": 20,
		"evaluate_during_training_steps" : 100,
    	"threshold": threshold,
    	"output_dir": output_dir,
    	"overwrite_output_dir": True,	
    	"best_model_dir": os.path.join(output_dir,'best_model'),
    	"manual_seed": manual_seed

	}
	return hyperparams

def calc_weighted_f1(labels,preds,threshold_setting,num_labels):
	if threshold_setting == 'default':
		threshold = 0.5
	elif threshold_setting == 'lower':
		threshold = 1.0 / num_labels
	y_pred = preds > threshold
	return sklearn.metrics.f1_score(labels,y_pred,average='weighted')

def calc_binary_f1(labels,preds,threshold_setting,num_labels):
	if threshold_setting == 'default':
		threshold = 0.5
	elif threshold_setting == 'lower':
		threshold = 1.0 / num_labels
	y_pred = preds > threshold
	return sklearn.metrics.f1_score(labels,y_pred,average='binary')

def calc_micro_f1(labels,preds,threshold_setting,num_labels):
	if threshold_setting == 'default':
		threshold = 0.5
	elif threshold_setting == 'lower':
		threshold = 1.0 / num_labels
	y_pred = preds > threshold
	return sklearn.metrics.f1_score(labels,y_pred,average='micro')

def calc_macro_f1(labels,preds,threshold_setting,num_labels):
	if threshold_setting == 'default':
		threshold = 0.5
	elif threshold_setting == 'lower':
		threshold = 1.0 / num_labels
	y_pred = preds > threshold
	return sklearn.metrics.f1_score(labels,y_pred,average='macro')


def calc_roc_auc(labels,preds):
	y_pred = preds > 0.5
	return sklearn.metrics.roc_auc_score(labels,y_pred,average='weighted')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lmpath",help='path to finetuned language model')
	parser.add_argument("--train-file",help='full path to training data file')
	parser.add_argument("--eval-file",help = 'full path to eval data file')
	parser.add_argument("--output-dir",help = 'output directory')

	parser.add_argument("--num-epochs",type=int,help='number of epochs to train')
	parser.add_argument("--thresh-setting",help='write default for 0.5 threshold for classification, and lower to normalize for number of labels')
	parser.add_argument("--device",type=int,help='which cuda device to run on')
	parser.add_argument("--manual-seed",type=int,help='seed for replicability')

	args = parser.parse_args()
	language_model_path = args.lmpath
	train_file = args.train_file
	eval_file = args.eval_file 
	output_dir = args.output_dir
	num_epochs = args.num_epochs
	thresh_setting = args.thresh_setting
	device = args.device
	manual_seed = args.manual_seed

	train_df,eval_df = load_data(train_file,eval_file)
	num_labels = len(train_df.columns) - 4 #everything but index,train,labels,id are classes
	model_args = configure_model_args(
		num_epochs=num_epochs,
		num_labels=num_labels,
		threshold_setting=thresh_setting,
		output_dir = output_dir, 
		manual_seed=manual_seed
		)

	model = MultiLabelClassificationModel('roberta',language_model_path, 
		num_labels=num_labels,
		args=model_args,
		cuda_device=device,
		) 
	model.train_model(train_df,
		output_dir =output_dir,
		eval_df=eval_df,
		#binary_f1 = lambda x,y: calc_binary_f1(x,y,thresh_setting,num_labels),
		macro_f1 = lambda x,y: calc_macro_f1(x,y,thresh_setting,num_labels),
		weighted_f1 = lambda x,y: calc_weighted_f1(x,y,thresh_setting,num_labels),
		micro_f1 = lambda x,y: calc_micro_f1(x,y,thresh_setting,num_labels),
		)
	result, model_outputs, wrong_predictions = model.eval_model(eval_df,
		output_dir =output_dir,
		#binary_f1 = lambda x,y: calc_binary_f1(x,y,tresh_setting,num_labels),
		macro_f1 = lambda x,y: calc_macro_f1(x,y,thresh_setting,num_labels),
		weighted_f1 = lambda x,y: calc_weighted_f1(x,y,thresh_setting,num_labels),
		micro_f1 = lambda x,y: calc_micro_f1(x,y,thresh_setting,num_labels),
		)

	preds_list,model_predict_outputs = model.predict(list(eval_df['text']))
	for i,elem in enumerate(list(eval_df['labels'])):
		print(list(eval_df['text'])[i],model_predict_outputs[i],'\t',elem)



if __name__ == "__main__":
	main()