import os 
import json
import gzip
import sys
import glob
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from os.path import abspath
from datetime import datetime

sys.path.insert(0, abspath('..'))
import parse_tweets


def load_user_ideologies(user_ideology_files):
	user_ideologies = {}
	for filename in user_ideology_files:
		with open(filename,'r') as f:
			reader = csv.reader(f)
			for row in reader:
				user_id = row[0]
				score = row[1]
				user_ideologies[user_id] = score
	return user_ideologies


def load_reaction_info(reaction_file):
	df = pd.read_csv(reaction_file,sep='\t',dtype=str)
	reactions_dict = defaultdict(dict)
	for i,row in df.iterrows():
		reactions_dict[row.id]['favorites'] = row.favorites
		reactions_dict[row.id]['retweets'] = row.retweets
	return reactions_dict




def get_metadata(tweet_file,country,user_ideology_dict,reactions_dict):
	all_tweets = parse_tweets.get_all_tweets([tweet_file],'',filter_retweet=False,filter_lang=False,filter_query=False)
	tweet_metadata = []
	for tweet_obj in all_tweets:
		tweet_features = get_tweet_features(tweet_obj)
		date_features = get_date_features(tweet_obj)
		user_features = get_user_features(tweet_obj,user_ideology_dict)
		op_features = get_op_features(tweet_obj,user_ideology_dict)
		reaction_features = get_reaction_features(tweet_obj,reactions_dict)
		all_features = {**tweet_features,**date_features,**user_features,**op_features,**reaction_features}
		all_features['country'] = country	
		tweet_metadata.append(all_features)

	return tweet_metadata


def get_ideology(user_ideology_dict,user_id):
	if user_id in user_ideology_dict:
		return float(user_ideology_dict[user_id])
	return np.nan


def get_length(tweet_obj):
	tweet_len = len(tweet_obj['text'])
	if 'extended_tweet' in tweet_obj:
		tweet_len = len(tweet_obj['extended_tweet']['full_text'])
	log_len = np.log(tweet_len)
	return log_len

def get_tweet_features(tweet_obj):
	tweet_features = {}
	tweet_features['id_str'] = str(tweet_obj['id_str'])
	tweet_features['log_chars'] = get_length(tweet_obj)
	tweet_features['has_hashtag'] = 1 if (len(tweet_obj['entities']['hashtags']) > 0) else 0
	tweet_features['has_url'] = 1 if (len(tweet_obj['entities']['urls']) > 0) else 0
	tweet_features['has_mention'] = 1 if (len(tweet_obj['entities']['user_mentions']) > 0) else 0
	tweet_features['is_reply'] = 1 if (tweet_obj['in_reply_to_status_id'] is not None) else 0
	tweet_features['is_quote_status'] = 1 if (tweet_obj['is_quote_status']) else 0
	return tweet_features


def get_date_features(tweet_obj):
	date_features = {}
	dtime = tweet_obj['created_at']
	new_datetime = datetime.strftime(datetime.strptime(dtime,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
	date_features['date'] = new_datetime.split('-')[2]
	date_features['month'] = new_datetime.split('-')[1]
	date_features['year'] = new_datetime.split('-')[0]
	return date_features


def get_user_features(tweet_obj,user_ideology_dict):	
	user_features = {}
	user = tweet_obj['user']
	user_features['log_statuses'] = np.log(int(user['statuses_count']))
	user_features['log_following'] = np.log(int(user['friends_count']+1))
	user_features['log_followers'] = np.log(int(user['followers_count']+1))
	user_features['is_verified'] = 1 if (user['verified']) else 0
	user_features['ideology'] = get_ideology(user_ideology_dict,user['id_str'])
	return user_features


def get_op_features(tweet_obj,user_ideology_dict):
	op_features = {}
	if tweet_obj['in_reply_to_user_id_str'] is not None:
		op_features['op_ideology'] = get_ideology(user_ideology_dict,tweet_obj['in_reply_to_user_id_str'])
		user_ideology = get_ideology(user_ideology_dict,tweet_obj['user']['id_str'])
		if (np.isnan(op_features['op_ideology']) == False) and (np.isnan(user_ideology) == False):
			op_features['opposed_ideology'] = 1 if (op_features['op_ideology']*user_ideology < 0) else 0
	return op_features


def get_reaction_features(tweet_obj,reaction_dict):
	tweet_id = tweet_obj['id_str']
	reaction_features = {}
	if tweet_id in reaction_dict:
		reaction_features['log_favorites'] = np.log(int(reaction_dict[tweet_id]['favorites'])+1)
		reaction_features['log_retweets'] = np.log(int(reaction_dict[tweet_id]['retweets'])+1)
	print(reaction_features)
	return reaction_features




def write_metadata_features(metadata,outfile): 
	keys = metadata[0].keys()
	with open(outfile, 'w', newline='')  as f:
	    dict_writer = csv.DictWriter(f, keys,delimiter='\t')
	    dict_writer.writeheader()
	    dict_writer.writerows(metadata)


def get_predicted_frames(predicted_frames_base,frame_types,year,country):
	df_list = []
	for frame_type in frame_types:
		filename = os.path.join(predicted_frames_base,frame_type,f'{country}_{year}_11-12-20.tsv')
		df_frames = pd.read_csv(filename,sep='\t').drop(columns=['Unnamed: 0'])
		df_list.append(df_frames)
		print(len(df_frames))
	return(pd.concat(df_list,axis=1))

def main():

	tweet_file_base = '/shared/2/projects/framing/data/immigration_tweets_by_country_07-16/'
	reaction_file_base = '/shared/2/projects/framing/data/num_fav_rt_07-16/'
	predicted_frames_base = '/shared/2/projects/framing/models/predict/'

	user_ideology_files = glob.glob('/shared/2/projects/framing/data/tweetscores_ideal_points/*')
	user_ideology_dict = load_user_ideologies(user_ideology_files)
	frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative']
	out_file = '/shared/2/projects/framing/results/full_datasheet_11-13-20.tsv'

	df_full_list = []

	for year in [2018,2019]:
		for country in ['EU','GB','US']:
			print(year,country)
			tweet_file = os.path.join(tweet_file_base,str(year),country+'.gz')
			reaction_file = os.path.join(reaction_file_base,f'{country}_{year}.tsv')
			reactions_dict = load_reaction_info(reaction_file)
			metadata = get_metadata(tweet_file,country,user_ideology_dict,reactions_dict)
			df_frames = get_predicted_frames(predicted_frames_base,frame_types, year,country)
			df_metadata = pd.DataFrame(metadata)
			print(len(df_frames),len(df_metadata))
			df = pd.concat([df_metadata,df_frames],axis=1)
			print(df)
			df_full_list.append(df)

	df_full = pd.concat(df_full_list,axis=0)
	df_full.to_csv(out_file,sep='\t')





if __name__ == "__main__":
	main()

