import os 
import random
import json
import gzip
import glob
import csv
from preprocess_text import tokenize,process_text
import re
import parse_tweets
import random
import numpy as np
import pandas as pd
from collections import defaultdict


def get_sample_tweets_metadata(tweet_filenames,sample_ids):
	all_tweets = parse_tweets.get_all_tweets(tweet_filenames,'',filter_retweet=False,filter_lang=False,filter_query=False)
	sample_tweets_metadata = []
	for tweet_obj in all_tweets:
		if tweet_obj['id_str'] in sample_ids:
			feature_dict = make_feature_dict_for_tweet(tweet_obj)
			sample_tweets_metadata.append(feature_dict)
	return sample_tweets_metadata


def make_feature_dict_for_tweet(tweet_obj):
	feature_dict = {}
	feature_dict['id_str'] = str(tweet_obj['id_str'])
	feature_dict['log_statuses'] = np.log(int(tweet_obj['user']['statuses_count']))
	feature_dict['num_chars'] = len(tweet_obj['text'])
	feature_dict['has_hashtag'] = 0
	feature_dict['has_url'] = 0
	feature_dict['has_mention'] = 0
	feature_dict['is_reply'] = 0
	if len(tweet_obj['entities']['hashtags']) > 0:
		feature_dict['has_hashtag'] = 1
	if len(tweet_obj['entities']['urls']) > 0:
		feature_dict['has_url'] = 1
	if len(tweet_obj['entities']['user_mentions']) > 0:
		feature_dict['has_mention'] = 1
	if tweet_obj['in_reply_to_status_id'] is not None:
		feature_dict['is_reply'] = 1
	return feature_dict

def write_metadata_features(metadata,outfile): 
	keys = metadata[0].keys()
	with open(outfile, 'w', newline='')  as f:
	    dict_writer = csv.DictWriter(f, keys,delimiter='\t')
	    dict_writer.writeheader()
	    dict_writer.writerows(metadata)



def main():
	tweet_filenames = glob.glob('/shared/2/projects/framing/data/immigration_tweets_by_country_07-16/*/*.gz')
	train_file = '/shared/2/projects/framing/data/dataset_07-23/train.json'
	outfile = '/shared/2/projects/framing/data/train_metadata_features_10-21.csv'

	df = pd.read_json(train_file,lines=True,dtype=str)
	sample_ids = set(df['id_str']) 

	sample_tweets_metadata = get_sample_tweets_metadata(tweet_filenames,sample_ids)
	write_metadata_features(sample_tweets_metadata,outfile)




if __name__ == "__main__":
	main()