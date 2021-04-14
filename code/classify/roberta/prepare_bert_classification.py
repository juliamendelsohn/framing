import pandas as pd 
import glob
from sklearn.preprocessing import MultiLabelBinarizer
import re
import os
import ast
import numpy as np


frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
frames = {}
frames['Issue-General'] = set(['Capacity and Resources', 'Crime and Punishment','Cultural Identity', 
'Economic','External Regulation and Reputation', 'Fairness and Equality','Health and Safety',
'Legality, Constitutionality, Jurisdiction', 'Morality and Ethics','Policy Prescription and Evaluation',
'Political Factors and Implications', 'Public Sentiment','Quality of Life', 'Security and Defense'])
frames['Issue-Specific'] = set(['Hero: Cultural Diversity','Hero: Integration', 'Hero: Worker',
'Threat: Fiscal', 'Threat: Jobs', 'Threat: National Cohesion','Threat: Public Order',
'Victim: Discrimination','Victim: Global Economy', 'Victim: Humanitarian', 'Victim: War'])
frames['Narrative'] = set(['Episodic','Thematic'])
frames['Issue-Specific-Combined'] = ['Hero','Threat','Victim']
frames['all_frames'] = set(sorted(list(frames['Issue-General'])+list(frames['Issue-Specific'])
+list(frames['Narrative'])+list(frames['Issue-Specific-Combined'])))


def load_labeled_data(filename):
	df = pd.read_csv(filename,sep='\t')
	df = df.loc[:, ~df.columns.str.match('Unnamed')]

	for frame_type in frame_types:
		if frame_type in df.columns:
			df[frame_type] = [ '[]' if x is np.NaN else x for x in df[frame_type] ]
	return df 

def add_combined_specific(df):
	spec_combined = []
	for i,entry in enumerate(list(df['Issue-Specific'])):
		combined_entry = '[]'
		if entry != '[]':
			combined_entry = str(list(set([s.split(':')[0] for s in ast.literal_eval(entry)])))
		spec_combined.append(combined_entry)
	df['Issue-Specific-Combined'] = spec_combined
	return df 


def format_multilabel_df(df,frame_type,outpath):
	print(df)
	new_df = pd.DataFrame()
	new_df['text'] = df['text']
	new_df['id'] = df['id_str']
	#new_df['frames'] = df[frame_type].apply(ast.literal_eval)
	new_cols = sorted(frames[frame_type])
	new_labels = {}
	for x in new_cols:
		new_labels[x] = []
	for index, row in df.iterrows():
		for frame_label in new_labels:
			if frame_label in row[frame_type]:
				new_labels[frame_label].append(1)
			else:
				new_labels[frame_label].append(0)
	for x in new_cols:
		new_df[x] = new_labels[x]
	labels = [new_df.loc[i,:].values.tolist()[2:] for i in range(len(new_df))]
	new_df['labels'] = labels
	outfile = os.path.join(outpath,frame_type + '.tsv')
	new_df.to_csv(outfile,'\t')
  

	# mlb = MultiLabelBinarizer()
	# binarized = pd.DataFrame(mlb.fit_transform(new_df.pop('frames')),columns=mlb.classes_)
	# new_df = pd.concat([new_df,binarized],sort=True,axis=1)
	# print(new_df.columns)
	

def add_issue_specific_combined_to_all_frames(issue_spec_combined_filename,all_frames_filename,outpath):
	df_issue = pd.read_csv(issue_spec_combined_filename,sep='\t')[['id','Hero','Threat','Victim']]
	df_all = pd.read_csv(all_frames_filename,sep='\t')
	df = df_all.merge(df_issue,on='id')
	df = df.loc[:, ~df.columns.str.match('Unnamed')]
	df = df.drop(columns=['labels'])
	labels = [df.loc[i,:].values.tolist()[2:] for i in range(len(df))]
	df['labels'] = labels
	outfile = os.path.join(outpath,'all_frames.tsv')
	df.to_csv(outfile,sep='\t')


def main():


	#labeled_data_path = '/home/juliame/framing/labeled_data/dataset_11-03-20/'
	labeled_data_path = '/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/'
	outpath_base = '/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta/'

	#for x in ['train','dev','test']:
	#for x in ['dev_liberal','dev_conservative','test_liberal','test_conservative',
	#'dev_US','dev_GB','dev_EU','test_US','test_GB','test_EU']:
	for x in ['eval_liberal','eval_conservative','eval_US','eval_GB','eval_EU']:
		subset = x.split('_')[1]
		file1 = os.path.join(labeled_data_path,f'dev_{subset}.tsv')
		file2 = os.path.join(labeled_data_path,f'test_{subset}.tsv')
		#filename = os.path.join(labeled_data_path, x + '.tsv')
		outpath = os.path.join(outpath_base,x)
		if not os.path.exists(outpath):
			os.mkdir(outpath)
		df1 = load_labeled_data(file1)
		df2 = load_labeled_data(file2)

		df1 = add_combined_specific(df1)
		df2 = add_combined_specific(df2)
		df = pd.concat([df1,df2]).reset_index()
		for frame_type in frame_types:
			format_multilabel_df(df,frame_type,outpath)

		issue_spec_combined_filename = os.path.join(outpath,'Issue-Specific-Combined.tsv')
		all_filename = os.path.join(outpath,'all_frames.tsv')
		add_issue_specific_combined_to_all_frames(issue_spec_combined_filename,all_filename,outpath)

if __name__ == "__main__":
	main()






