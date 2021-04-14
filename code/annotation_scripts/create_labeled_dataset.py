import pandas as pd 
import glob
import json
import os
import numpy as np
from functools import reduce
from collections import Counter

#split is either train dev or test
def load_annotated_text(annotated_data_path,start_year,end_year,countries,split):
	years = [str(y) for y in range(start_year,end_year+1)]
	text_df_list = []

	for year in years:
		for country in countries:
			filename = os.path.join(annotated_data_path,f'{year}-{country}-{split}.json')
			df = pd.read_json(filename,lines=True,dtype=str)
			df['year'] = year
			df['country'] = country
			text_df_list.append(df)

	text_df = pd.concat(text_df_list).reset_index().drop(columns=['index'])
	return text_df


def load_ideology_for_annotated_text(ideology_file,annotated_ids):
	annotated_ideologies = []

	with open(ideology_file,'r') as f:
		tweet_ideology = json.load(f)

	mean_ideology = np.mean([float(v) for v in tweet_ideology.values()])
	for tweet_id in annotated_ids:
		if tweet_id in tweet_ideology:
			entry = {}
			entry['id_str'] = tweet_id
			entry['ideology'] = tweet_ideology[tweet_id]
			score = float(tweet_ideology[tweet_id])

			if score < 0:
				entry['libcon_raw'] = 'liberal'
			elif score > 0:
				entry['libcon_raw'] = 'conservative'
			if score < mean_ideology:
				entry['libcon_norm'] = 'liberal'
			elif score > mean_ideology:
				entry['libcon_norm'] = 'conservative'
			annotated_ideologies.append(entry)
	df = pd.DataFrame(annotated_ideologies)
	return df


def load_annotations(annotation_output,frame_types):
	annotations = []
	with open(annotation_output,'r') as f:
		for line in f:
			example = json.loads(line)
			entry = {}
			entry['id_str'] = str(example['id'])
			entry['all_frames'] = []
			for frame_type in frame_types:
				entry[frame_type] = []
				if frame_type in example['annotation']:
					entry[frame_type] = [l for l in example['annotation'][frame_type] if l != 'None']
				entry['all_frames'] += entry[frame_type]
			annotations.append(entry)
	return pd.DataFrame(annotations)

def fill_incomplete_annotation(text_df,annotations):

	set1 = set(text_df['id_str'].astype(str))
	set2 = set(annotations['id_str'].astype(str))
	incompleted_ids = set1 - set2
	print(incompleted_ids)
	#######
	# Incompleted ids {'1079396885163520000', '1000453898560294912'}
	#'1079396885163520000': 
		#Morality and Ethics,  'Security and Defense', Economic, Policy Prescription and Evaluation
		#Victim: Humanitarian'
		#Thematic
	#1000453898560294912
		#'Morality and Ethics', 'Security and Defense', 'Public Sentiment']
		#'Victim: Humanitarian'
		#Episodic, Thematic
	# This part is hard-coded annotations that didn't go through potato
	###
	
	new_entry_1 = {'id_str': '1079396885163520000',
				'all_frames': ['Morality and Ethics',  'Security and Defense', 'Economic', 'Policy Prescription and Evaluation','Victim: Humanitarian','Thematic'],
				'Issue-General': ['Morality and Ethics',  'Security and Defense', 'Economic', 'Policy Prescription and Evaluation'],
				'Issue-Specific': ['Victim: Humanitarian'],
				'Narrative': ['Thematic']}

	new_entry_2 = {'id_str': '1000453898560294912',
			'all_frames': ['Morality and Ethics', 'Security and Defense', 'Public Sentiment','Victim: Humanitarian','Episodic', 'Thematic'],
			'Issue-General': ['Morality and Ethics', 'Security and Defense', 'Public Sentiment'],
			'Issue-Specific': ['Victim: Humanitarian'],
			'Narrative': ['Episodic', 'Thematic']}


	new_annotations = annotations.append(pd.DataFrame([new_entry_1,new_entry_2]))
	return new_annotations
	

def combine_labels_metadata(text_df,ideology_df,annotations_df):
	df = pd.merge(text_df,annotations_df,on='id_str')
	df = pd.merge(df,ideology_df,on='id_str',how='left')
	return df


def main():
	#annotation_order_file = "/home/juliame/potato/annotation_output_train/Julia_Mendelsohn/annotation_order.txt"
	annotation_output = "/home/juliame/framing/labeled_data/eval_annots.jsonl"
	annotated_data_path = "/home/juliame/potato/data/07-23-20/by_year_country"
	ideology_file = "/shared/2/projects/framing/data/tweet-ideology-07-16.json"
	ideology_path_lib_con = "/shared/2/projects/framing/data/us_tweets_by_ideology_07-16"
	out_path = '/home/juliame/framing/labeled_data/dataset_11-03-20'

	start_year = 2018
	end_year = 2019
	countries = ['EU','US','GB']

	annotations_df = load_annotations(annotation_output,['Issue-General','Issue-Specific','Narrative'])
	ideology_df = load_ideology_for_annotated_text(ideology_file,list(annotations_df['id_str']))


	for split in ['dev','test']:
		text_df = load_annotated_text(annotated_data_path,start_year,end_year,countries,split)
		dataset = combine_labels_metadata(text_df,ideology_df,annotations_df)
		out_file = os.path.join(out_path,split + '.tsv')
		dataset.to_csv(out_file,sep='\t')
	




if __name__ == "__main__":
	main()

