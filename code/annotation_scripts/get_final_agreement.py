import pandas as pd 
import glob
import json
import os
import numpy as np
from collections import Counter
import nltk
from nltk.metrics import masi_distance, agreement
from functools import reduce



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


def get_dataset_by_coder(annotation_output_path,coder):
	annotation_output_file = os.path.join(annotation_output_path,coder + '_all.jsonl')
	frame_types = ['Issue-General','Issue-Specific','Narrative']
	annotations = load_annotations(annotation_output_file,frame_types)
	return annotations


def combine_multiple_coders(annotation_output_path,coders):
	all_coders_annotations = []
	for coder in coders:
		df = get_dataset_by_coder(annotation_output_path,coder)
		df = pd.melt(df,id_vars='id_str')
		df.columns = ['id_str','frame_type','frames']
		all_coders_annotations.append(df)

	new_df = reduce(lambda l,r: pd.merge(l,r,on=['id_str','frame_type'],how='outer'), all_coders_annotations)
	new_df.columns = ['id_str','frame_type'] + [f'labels-{coder}' for coder in coders]
	return new_df

def format_labels(combined_df,coders,frame_type):
	df = combined_df[combined_df['frame_type']==frame_type].dropna()
	all_labels = []

	for col in df.columns:
		df[col] = df[col].apply(lambda y: np.nan if type(y) == list and len(y)==0 else y)
	df = df.dropna()

	for coder_num,coder in enumerate(coders):
		for i,row in enumerate(df.iterrows()):
			labels = frozenset(sorted(df.iloc[i][f'labels-{coder}']))
			all_labels.append((coder_num,i,labels))
	return all_labels


# #distance metric is either masi_distance or binary
def overall_agreement(combined_df,coders,frame_types,distance_metric=masi_distance):
	results = []
	for frame_type in frame_types:
		all_labels = format_labels(combined_df,coders,frame_type)
		task = agreement.AnnotationTask(data=all_labels,distance=distance_metric)
		results.append((frame_type,task.alpha()))
	agree_df = pd.DataFrame(results,columns=['Frame Type','Alpha'])
	agree_df['Coder'] = str(coders)
	return agree_df


def compare_with_consensus(coder,consensus_file,frame_types,annotation_output_path_base):
	consensus_annots = load_annotations(consensus_file,frame_types)
	consensus_df = pd.melt(consensus_annots,id_vars='id_str')
	consensus_df.columns = ['id_str','frame_type','frames']


	coder_dfs = []
	for annot_file in glob.glob(os.path.join(annotation_output_path_base,'*',coder+'_all.jsonl')):
		coder_dfs.append(load_annotations(annot_file,frame_types))
	coder_df_full = pd.concat(coder_dfs)
	coder_df_full = pd.melt(coder_df_full,id_vars='id_str')
	coder_df_full.columns = ['id_str','frame_type','frames']



	combined_df = pd.merge(consensus_df,coder_df_full,on=['id_str','frame_type'],how='right')
	combined_df.columns = ['id_str','frame_type','labels-consensus',f'labels-{coder}']
	agreement_df = overall_agreement(combined_df,(coder,'consensus'),frame_types)
	return agreement_df




def main():

	annotation_output_path_base = '/home/juliame/framing/labeled_data/annotations_by_pair'
	consensus_file = '/home/juliame/framing/labeled_data/eval_annots.jsonl'
	coder_pairs = [(c,'julia') for c in ['ceren','david','anoop','shiqi']]
	frame_types = ['Issue-General','Issue-Specific','Narrative']

	full_agreement = []

	for coder_pair in coder_pairs:
		annotation_output_path = os.path.join(annotation_output_path_base,coder_pair[0] + '_' + coder_pair[1])
		combined_df = combine_multiple_coders(annotation_output_path,coder_pair)
		agreement = overall_agreement(combined_df.dropna(),coder_pair,frame_types)
		full_agreement.append(agreement)

	for coder in ['ceren','david','shiqi','anoop','julia']:
		consensus_agree_df = compare_with_consensus(coder,consensus_file,frame_types,annotation_output_path_base)
		full_agreement.append(consensus_agree_df)


	full_agree_df = pd.concat(full_agreement)
	full_agree_df.to_csv('/home/juliame/framing/labeled_data/eval_interannotator_agreement.tsv',sep='\t')

	




if __name__ == "__main__":
	main()