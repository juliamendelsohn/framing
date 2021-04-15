import pandas as pd 
import glob
import json
import os
import numpy as np
from collections import Counter
import nltk
from nltk.metrics import masi_distance, agreement
from functools import reduce

"""
This script takes in the annotation output files for given pairs of coders from Potato,
calculates interannotator agreement on sample of tweets (by index number),
and outputs tsv file of all tweets where annotators disagreed on any frame type 
(issue-generic, issue-specific, or narrative). We used these output tsv files 
to discuss disagreements in consensus-building meetings. 
"""

def load_annotated_text(annotated_data_file):
	df = pd.read_json(annotated_data_file,lines=True,dtype=str)
	return df


def load_annotations(annotation_output,frame_types):
	annotations = []
	with open(annotation_output,'r') as f:
		for line in f:
			example = json.loads(line)
			entry = {}
			entry['id_str'] = str(example['id'])
			for frame_type in frame_types:
				entry[frame_type] = []
				if frame_type in example['annotation']:
					entry[frame_type] = example['annotation'][frame_type]
			annotations.append(entry)
	return pd.DataFrame(annotations).fillna('[None]')


# Tweet text needed for consensus-building
def combine_labels_text(text_df,annotations,frame_types):
	df = pd.merge(text_df,annotations,on='id_str',how='outer').dropna()
	id_order = df['id_str']
	df = pd.melt(df, id_vars=['id_str','text'], value_vars=frame_types)
	df['id_str'] = pd.Categorical(df['id_str'], id_order)
	df = df.sort_values("id_str").reset_index().drop(columns=['index'])
	df.columns = ['id_str','text','frame_type','labels']
	return df

def get_dataset_by_coder(annotation_output_path,annotated_data_file,coder):
	annotation_output_file = os.path.join(annotation_output_path,coder,'annotated_instances.jsonl')
	frame_types = ['Issue-General','Issue-Specific','Narrative']
	text_df = load_annotated_text(annotated_data_file)
	annotations = load_annotations(annotation_output_file,frame_types)
	dataset = combine_labels_text(text_df,annotations,frame_types)
	return dataset


def combine_multiple_coders(annotation_output_path,annotated_data_file,coders):
	all_coders_annotations = []
	for coder in coders:
		df = get_dataset_by_coder(annotation_output_path,annotated_data_file,coder)
		df.columns = ['id_str','text','frame_type',f'labels-{coder}']
		all_coders_annotations.append(df)

	new_df = reduce(lambda l,r: pd.merge(l,r,on=['id_str','text','frame_type'],how='outer'), all_coders_annotations)
	return new_df

def get_disagreements_only(df,coders):
	df.columns = df.columns.str.replace("-", "_")
	cols = [f'labels_{coder}' for coder in coders]
	disagree_df = df.copy()
	for i,col in enumerate(cols):
		if i > 0:
			query = cols[i] + ' != ' + cols[i-1]
			disagree_df = disagree_df.query(query)
	return disagree_df

# Function to format labels from coders' dataframes for agreement calculation
def format_labels(combined_df,coders,frame_type):
	df = combined_df.dropna()
	df = combined_df[combined_df['frame_type']==frame_type].dropna()
	all_labels = []
	for coder_num,coder in enumerate(coders):
		for tweet_num,tweet in enumerate(df.iterrows()):
			labels = frozenset(sorted(df.iloc[tweet_num][f'labels-{coder}']))
			all_labels.append((coder_num,tweet_num,labels))
	return all_labels


# #distance metric is either masi_distance or binary
def overall_agreement(combined_df,coders,frame_types,distance_metric=masi_distance):
	results = []
	for frame_type in frame_types:
		all_labels = format_labels(combined_df,coders,frame_type)
		task = agreement.AnnotationTask(data=all_labels,distance=distance_metric)
		results.append((frame_type,task.alpha()))
	return pd.DataFrame(results,columns=['Frame Type','Alpha'])


def main():
	# Create directories for each annotator pair
	annotation_output_path = "/home/juliame/potato/annotation_output_eval6/"
	annotated_data_file = "/home/juliame/potato/data/07-23-20/eval_09-29-20/eval6.json"
	combined_output_path = "/home/juliame/framing/labeled_data/eval6_annotations/"
	coders = ['Julia_Mendelsohn','Anoop_Kotha'] # Change for relevant pair of coders
	frame_types = ['Issue-General','Issue-Specific','Narrative']
	
	if not os.path.exists(combined_output_path):
		os.mkdir(combined_output_path)
	
	# New files created for each consensus-building session
	date = '10-29-20'
	start_num=0
	end_num=75  # Goal was to do 75 tweets per meeting, but annotators completed varying amounts each time
	combined_outfile = os.path.join(combined_output_path,f'annotations_{date}_{start_num}_{end_num}.tsv')
	agreement_outfile = os.path.join(combined_output_path,f'agreement_{date}_{start_num}_{end_num}.tsv')
	disagree_outfile = os.path.join(combined_output_path,f'disagree_{date}_{start_num}_{end_num}.tsv')

	combined_df = combine_multiple_coders(annotation_output_path,annotated_data_file,coders)
	# Just get agreement scores and disagreement for one session at a time
	combined_df = combined_df[len(frame_types)*start_num:len(frame_types)*(end_num)]

	# Handles missing data better
	for col in combined_df.columns:
		combined_df[col] = combined_df[col].apply(lambda y: np.nan if type(y) == list and len(y)==0 else y)


	agreement = overall_agreement(combined_df.dropna(),coders,frame_types)
	disagree = get_disagreements_only(combined_df,coders)

	combined_df.to_csv(combined_outfile,sep='\t')
	agreement.to_csv(agreement_outfile,sep='\t')
	disagree.to_csv(disagree_outfile,sep='\t')
	




if __name__ == "__main__":
	main()