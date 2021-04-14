import nltk
from nltk.metrics import masi_distance, agreement
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict,Counter
from itertools import combinations
import json
sns.set_palette('colorblind')
import sys

# Function to load annotations for each annotator
# Path,date,and list of coders needed for the files' full path
# Returns list of Pandas dataframes, where each dataframe is one annotator's spreadsheet
def load_annotations(path,dates,coders):
	all_dfs = []
	if type(dates)==str:
		dates = [dates]
	for coder in coders:
		combined_df = pd.DataFrame()
		for date in dates:
			df = read_annotations(path,date,coder)
			if combined_df.empty:
				combined_df = df
			else:
				combined_df =  pd.concat([combined_df,df],sort=False).reset_index(drop=True)
		all_dfs.append(combined_df)
	return all_dfs

#returns a single dataframe corresponding to one annotation file
def read_annotations(path,date,coder):
	tsv_filename = os.path.join(path,date,coder) + '.tsv'
	jsonl_filename = os.path.join(path,date,coder,'annotated_instances.jsonl')
	if os.path.exists(jsonl_filename):
		tweets = []
		with open(jsonl_filename,'r') as f:
			for line in f:
				d = json.loads(line)
				tweet = {}
				tweet['id'] = d['id']
				tweet['Text'] = get_text(path,date,d['id'])
				for frame_type in d['annotation']:
					tweet[frame_type] = ','.join([get_shorthand(frame) for frame in d['annotation'][frame_type]])
				tweets.append(tweet)
		return pd.DataFrame(tweets).fillna('none')
	elif os.path.exists(tsv_filename):
		df = pd.read_csv(tsv_filename,sep='\t',error_bad_lines=False,quoting=3)
		df.columns = ['id','Text','Issue-General','Issue-Specific','Narrative']
		return df.fillna('none')

def get_text(path,date,tweet_id):
	datafile = os.path.join(path,date,'data.json')
	with open(datafile,'r') as f:
		for line in f:
			d = json.loads(line)
			if d['id_str'] == str(tweet_id):
				return d['text']


#returns shorthand label for each frame used in original annotations
def get_shorthand(frame):
	shorthand = {
	"Capacity and Resources": 'resources',
	"Morality and Ethics":'morality',
	"Fairness and Equality": 'fairness',
	"Legality, Constitutionality, Jurisdiction": 'legality',
	"Crime and Punishment": 'crime',	
	"Security and Defense": 'security',						 
	"Political Factors and Implications": 'political',
	"Policy Prescription and Evaluation": 'policy',
	"External Regulation and Reputation": 'external',
	}
	if frame in shorthand:
		return shorthand[frame]
	else:
		return frame.lower().replace(': ','/')


def load_consensus(path,dates):
	return load_annotations(path,dates,['consensus'])[0]

#Puts labels into the form of (coder_num,tweet_num,frame_set), returns list of labels
#Requires the list of dataframes and frame_type (Issue-general,Immigration-specific,Narrative)
#Coder_num just differentiates annotators from each other for the agreement calculation
#Tweet_num differentiates tweets from each other for the agreement calculation (to match responses to the same tweet)
#Assigned frames converted from comma-separated to frozen set
def format_labels(df_list,frame_type):
	all_labels = []
	for coder_num,df in enumerate(df_list):
		labels = []
		for tweet_num,tweet in enumerate(df[frame_type]):
			labels.append((coder_num,tweet_num,frozenset(sorted(tweet.split(',')))))
		all_labels += labels
	return all_labels


#Input is list of annotation DataFrames (from load_annotation)
#distance metric is either masi_distance or binary
def overall_agreement(df_list,frame_types,distance_metric=masi_distance,convert_mixed=True):
	results = []
	for frame in frame_types:
		if frame == 'Narrative' and convert_mixed==True:
			new_df_list = convert_narrative_mixed_to_both(df_list)
			all_labels = format_labels(new_df_list,frame)
		else:
			all_labels = format_labels(df_list,frame)
		task = agreement.AnnotationTask(data=all_labels, distance=distance_metric)
		results.append((frame,task.alpha()))
	return pd.DataFrame(results,columns=['Frame Type','Alpha'])


def convert_narrative_mixed_to_both(df_list):
	new_df_list = df_list.copy()
	for df in new_df_list:
		converted = [x if x.split('/')[0] !='mixed' else 'episodic,thematic' for x in df['Narrative']]
		df['Narrative'] = converted
	return new_df_list

def collapse_specific_subcat(df_list):
	new_df_list = df_list.copy()
	for df in new_df_list:
		converted = []
		for elem in df['Issue-specific']:
			combined = ','.join([x.split('/')[0] for x in elem.split(',')])
			converted.append(combined)
		df['Issue-specific'] = converted
	return new_df_list


def compare_over_samples(path,dates,coders,frame_types,distance_metric=masi_distance,convert_mixed=True,collapse_subcat=False):
	df = pd.DataFrame()
	for date in dates:
		df_list = load_annotations(path,date,coders)
		if collapse_subcat:
			df_list = collapse_specific_subcat(df_list)
		new_df = overall_agreement(df_list,frame_types,distance_metric,convert_mixed)
		new_df['Date'] = date
		df = pd.concat([df,new_df])
	return df


def leave_one_out(path,dates,coders,frame_types,distance_metric=masi_distance,convert_mixed=True):
	df = pd.DataFrame()
	for coder in coders:
		others = [c for c in coders if c != coder]
		df_list = load_annotations(path,dates,others)
		new_df = overall_agreement(df_list,frame_types,distance_metric,convert_mixed)
		new_df['Left out'] = coder
		df = pd.concat([df,new_df])
	return df



def individual_vs_pair_agreement(path,dates,coders,pair_coders,frame_types,distance_metric=masi_distance,convert_mixed=True):
	indiv_df_list = load_annotations(path,dates,coders)
	pair_df_list = load_annotations(path,dates,pair_coders)
	scores_ind = overall_agreement(indiv_df_list,frame_types,distance_metric,convert_mixed)
	scores_pair = overall_agreement(pair_df_list,frame_types,distance_metric,convert_mixed)
	scores_ind['Group'] = 'Individual'
	scores_pair['Group'] = 'Pair'
	df = pd.concat([scores_ind,scores_pair])
	return df


def pairwise_agreement(path,dates,coders,frame_types,distance_metric=masi_distance,convert_mixed=True):
	coder_pairs = list(combinations(coders,2))
	df = pd.DataFrame()
	for pair in coder_pairs:
		df_list = load_annotations(path,dates,pair)
		new_df = overall_agreement(df_list,frame_types,distance_metric,convert_mixed)
		new_df['Pair'] = str(pair)
		df = pd.concat([df,new_df])
	return df

def agreement_combined_subcat(path,dates,coders):
	df_list = load_annotations(path,dates,coders)
	frame_types = ['Issue-specific']
	scores_full = compare_over_samples(path,dates,coders,frame_types,collapse_subcat=False)
	scores_combined = compare_over_samples(path,dates,coders,frame_types,collapse_subcat=True)
	scores_full['Categories'] = 'Separate'
	scores_combined['Categories'] = 'Combined'
	df = pd.concat([scores_full,scores_combined])
	return df 



def compare_to_consensus(path,dates,coders,frame_types,distance_metric=masi_distance,convert_mixed=True):
	coder_pairs = [(coder,'consensus') for coder in coders]
	results = []
	df = pd.DataFrame()
	for pair in coder_pairs:
		df_list = load_annotations(path,dates,pair)
		new_df = overall_agreement(df_list,frame_types,distance_metric,convert_mixed)
		new_df['Coder'] = pair[0]
		df = pd.concat([df,new_df])
	return df



def get_consensus_distributions(consensus,frame_type,combine_subcat=False):
	counts = Counter()
	frame_sets = consensus[frame_type]
	for frame_set in frame_sets:
		for frame in frame_set.split(','):
			if combine_subcat:
				counts[frame.split('/')[0]] += 1
			else:
				counts[frame] += 1
	df = pd.DataFrame(counts.most_common(),columns=['Frame','Count'])
	sns.barplot(x='Count',y='Frame',data=df)
	return df

		
# Function to identify where annotators disagree
# Given a specific frame type and all dataframes, returns list of (text,possible frames) where people disagree
def get_all_labels(path,dates,coders,frame_types):
	df_list = load_annotations(path,dates,coders)
	all_labels = defaultdict(lambda:defaultdict(list))
	for df in df_list:
		for index,row in df.iterrows():
			for frame_type in frame_types:
				text = row['Text']
				frames = sorted(row[frame_type].split(','))
				if frames not in all_labels[text][frame_type]:
					all_labels[text][frame_type].append(frames)
	return all_labels


def get_disagreements(path,dates,coders,frame_types):
	all_labels = get_all_labels(path,dates,coders,frame_types)
	disagreements = []
	for text in all_labels:
		for frame_type in all_labels[text]:
			if len(all_labels[text][frame_type]) > 1: #means there's disagreement
				candidates = []
				for response in all_labels[text][frame_type]:
					candidates += response
				candidates = list(set(candidates))
				entry = tuple([text,frame_type] + candidates)
				disagreements.append(entry)
	return disagreements

def write_disagreements(disagreements,output_file):
	df = pd.DataFrame(disagreements)
	df.to_csv(output_file,sep='\t')



def get_candidate_set(disagreements,frame_type):
	cand = {}
	for tweet in disagreements:
		if tweet[1] == frame_type:
			text = tweet[0]
			cand[text] = tweet[2:]
	return cand


def get_mislabeled(consensus,disagreements,frame_type):
	cand = get_candidate_set(disagreements,frame_type)
	mislabels = defaultdict(lambda:defaultdict(int))
	for i,row in consensus.iterrows():
		text = row['Text']
		if text in cand:
			true_frames = row[frame_type].split(',')
			for frame in true_frames:
				for label in cand[text]:
					if label not in true_frames:
						mislabels['true_' +frame][label] += 1
	return mislabels

def create_mislabeled_matrix(consensus,disagreements,frame_type,norm=False):
	mislabel = get_mislabeled(consensus,disagreements,frame_type)
	confusion = pd.DataFrame(mislabel).fillna(0).T
	if frame_type == 'Issue-General':
		column_order = ['economic','resources','morality','fairness','legality','crime','security',
		'health and safety','quality of life','cultural identity','public sentiment','political','policy','external','none']
	elif frame_type == 'Issue-specific':
		column_order = ['victim/global economy','victim/humanitarian','victim/war','victim/discrimination',
						'hero/cultural diversity','hero/integration','hero/workers',
						'threat/jobs','threat/public order','threat/fiscal','threat/national cohesion','none']
	elif frame_type == "Narrative":
		column_order = ['episodic','thematic','mixed','none']

	row_order = ['true_' + x for x in column_order]
	confusion = confusion.reindex(columns=column_order)
	confusion = confusion.reindex(row_order).fillna(0)
	if norm:
		confusion = confusion.div(confusion.sum(axis=1), axis=0)
	cmap = sns.cubehelix_palette(light=1, as_cmap=True)
	sns.heatmap(confusion, annot=False,cmap=cmap)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	




#compare_over = Date for comparing over samples 
#compare_over = Group for comparing individual vs pair
#compare_over = Left out for comparing leave one out
def plot_agreement(df,title,compare_over):
	sns.barplot(x='Frame Type',y='Alpha',data=df,hue=compare_over)
	plt.legend(loc='center left', bbox_to_anchor=(1, .5))
	plt.xticks(rotation=10)
	plt.title(title)


# input dataframe has pairwise agreement scores 
def plot_pairwise_agreement(df,frame_type):
	minidf = df[df['Frame Type']==frame_type].sort_values(['Alpha'],ascending=False)
	sns.barplot(x='Alpha',y='Pair',data=minidf)
	plt.title(f'Pairwise agreement for {frame_type} frames')

def plot_specific_agreement_combined_vs_separate(df,title):
	sns.barplot(x='Date',y='Alpha',data=df,hue='Categories')
	plt.legend(loc='center left', bbox_to_anchor=(1, .5))
	plt.title(title)


def load_frames():
	frames = {}
	frames['Issue-general'] = ['economic','resources','morality','fairness','legality','crime','security',
			  'health and safety','quality of life','cultural identity','public sentiment',
			  'political','policy','external']
	frames['Issue-specific'] = ['victim/global economy','victim/humanitarian','victim/war','victim/discrimination',
			'hero/cultural diversity','hero/integration','hero/workers',
			'threat/jobs','threat/public order','threat/fiscal','threat/national cohesion']
	frames['Narrative'] = ['episodic','thematic','mixed/prominent','mixed/report','mixed/personal']
	return frames


def simulate_frame_deletion(frame_type,frames_to_delete,df_list):
	new_df_list = []
	for coder_num,df in enumerate(df_list):
		new_df = df.copy()
		new_annotations = []
		for annotation in list(df[frame_type]):
			annotations = set(annotation.split(','))
			modified = list(set(annotations).difference(set(frames_to_delete)))
			if len(modified) == 0:
				modified = ['none']
			new_annotations.append(','.join(modified))
		new_df[frame_type] = new_annotations
		new_df_list.append(new_df)
	return new_df_list 

def simulate_frame_combining(frame_type,frames_to_combine,df_list):
	new_df_list = []
	for coder_num,df in enumerate(df_list):
		new_df = df.copy()
		new_annotations= []
		for annotation in list(df[frame_type]):
			modified = annotation
			for frame in frames_to_combine[1:]:
				modified = modified.replace(frame,frames_to_combine[0])
			modified = set(modified.split(','))
			new_annotations.append(','.join(modified))
		new_df[frame_type] = new_annotations
		new_df_list.append(new_df)
	return new_df_list

def get_combos(frames, max_num,min_num=0):
	combos = []
	for i in range(min_num,max_num+1):
		combos += list(combinations(frames,i))
	return combos

def simulate_all_combining(all_dfs):
	all_dfs = load_annotations(path,dates,coders)
	frames = load_frames()
	results = []
	for frame_type in frames.keys():
		baseline = get_score(all_dfs,frame_type)
		combos = get_combos(frames[frame_type],2,min_num=2)
		for c in combos:
			new_dfs = simulate_frame_combining(frame_type,list(c),all_dfs)
			score = get_score(new_dfs,frame_type) - baseline
			results.append((frame_type,c,score))
	return pd.DataFrame(results,columns=['Frame Type','Combined Frames','Improvement']).sort_values(['Frame Type','Improvement'],ascending=False)

def simulate_all_deleting(all_dfs):
	frames = load_frames()
	results = []
	for frame_type in frames.keys():
		baseline = get_score(all_dfs,frame_type)
		combos = get_combos(frames[frame_type],1)
		for c in combos:
			new_dfs = simulate_frame_deletion(frame_type,list(c),all_dfs)
			score = get_score(new_dfs,frame_type) - baseline
			results.append((frame_type,c,score))
	return pd.DataFrame(results,columns=['Frame Type','Deleted Frames','Improvement']).sort_values(['Frame Type','Improvement'],ascending=False)       



#USAGE: 
#frame_type = 'Issue-general'
#frames_to_delete = ['quality of life','external','public sentiment','legality']
#frames_to_combine = [['economic','resources'],['policy','political'],['fairness','morality']]
#simulate_changes(df_list,frame_type,frames_to_delete,frames_to_combine)
def simulate_changes(all_dfs,frame_type,frames_to_delete,frames_to_combine):
	print("INITIAL",get_score(df_list,frame_type))
	df_delete_list = simulate_frame_deletion(frame_type,frames_to_delete,df_list)
	print("DELETE",frames_to_delete,get_score(df_delete_list,frame_type))
	df_list_prev = df_delete_list.copy()
	df_list_temp = df_delete_list.copy()
	for combo in frames_to_combine:
		df_list_temp = simulate_frame_combining(frame_type,combo,df_list_prev)
		df_list_prev = df_list_temp.copy()
		print("COMBINE",combo,get_score(df_list_temp,frame_type))
	final_df_list = df_list_temp.copy()
	print("FINAL",get_score(final_df_list,frame_type))

















