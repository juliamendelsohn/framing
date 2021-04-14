import os
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from simpletransformers.classification import MultiLabelClassificationModel



def load_data(train_path,predict_file,frame_type):
	all_data = defaultdict(list)
	train_file = os.path.join(train_path,frame_type + '.tsv')
	df_train = pd.read_csv(train_file,sep='\t',converters={'labels': literal_eval})

	df_predict = pd.read_csv(predict_file,sep='\t',header=None)
	df_predict.columns = ['user_name','id_str','text']
	all_data['X_train'] = df_train.text
	all_data['X_predict'] = df_predict.text

	all_data['y_train'] = 2.drop(columns=['Unnamed: 0','text','id','labels'])
	all_data['labels'] = list(all_data['y_train'].columns)

	return all_data

def load_model(model_path,device):
	clf = MultiLabelClassificationModel('roberta',model_path,cuda_device=device) 
	return clf


def run_model(clf,data):
	X_train = data['X_train']
	y_train = data['y_train']
	X_predict = data['X_predict']
		
	y_pred = clf.predict(X_predict)
	y_pred = pd.DataFrame((y_pred[0]))
	y_pred.columns=y_train.columns

	print(y_pred)

	return y_pred
	

def write_predictions(y_pred,data,out_file):
	# X_predict = data['X_predict']
	# print(X_predict)
	# print(y_pred)
	# df = pd.concat([X_predict,y_pred],axis=1)
	y_pred.to_csv(out_file,sep='\t')
	#df.to_csv(out_file,sep='\t')


def main():
	train_data_path = '/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta/train'
	full_data_path = '/shared/2/projects/framing/data/immigration_tweets_by_country_07-16/'
	model_base_path = '/shared/2/projects/framing/models/classify/'
	out_base_path = '/shared/2/projects/framing/models/predict/'
	frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']

	years = [2018,2019]
	countries = ['EU','GB','US']
	device = 7


	#frame_types = ['Narrative']
	#frame_types = ['Issue-General']
	#frame_types = ['Issue-Specific']
	frame_types = ['Issue-Specific-Combined']


	for frame_type in frame_types:
		for year in years:
			for country in countries:
				model_path = os.path.join(model_base_path,frame_type,'11-03-20_60_epochs_default_thresh_12_seed')
				data_file = os.path.join(full_data_path,str(year),country+'.tsv')
				frame_type_path = os.path.join(out_base_path,frame_type)
				if not os.path.exists(frame_type_path):
					os.mkdir(frame_type_path)
				out_file = os.path.join(out_base_path,frame_type,f'{country}_{year}_11-12-20.tsv')
				data = load_data(train_data_path,data_file,frame_type)
				model = load_model(model_path,device)
				y_pred = run_model(model,data)
				write_predictions(y_pred,data,out_file)


if __name__ == "__main__":
	main()