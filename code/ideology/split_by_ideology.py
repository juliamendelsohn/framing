import pandas as pd 
import json
import csv
import os 
import parse_tweets
import glob
import numpy as np
from collections import defaultdict



def split_by_ideology(all_tweets,ideology_dict,norm=False):
	result = defaultdict(list)
	threshold = 0
	if norm:
		threshold = np.mean([float(v) for v in ideology_dict.values()])

	for tweet in all_tweets:
		tid = tweet['id_str']
		if tid in ideology_dict and float(ideology_dict[tid]) < threshold:
			result['liberal'].append(tweet)
		elif tid in ideology_dict and float(ideology_dict[tid]) > threshold:
			result['conservative'].append(tweet)
	return result
	

def write_separate_ideology_files(split_tweets,out_path,suffix=''):
	for ideology in split_tweets:
		outfile = os.path.join(out_path,ideology + suffix + '.gz')
		parse_tweets.write_tweets(outfile,split_tweets[ideology])



def main():
	data_dir = '/shared/2/projects/framing/data/'
	tweet_base_path = os.path.join(data_dir,'immigration_tweets_by_country_07-16/')
	us_tweet_files = glob.glob(os.path.join(tweet_base_path,'*','US.gz'))
	tweet_ideology_file = os.path.join(data_dir,'tweet-ideology-07-16.json')
	out_path = os.path.join(data_dir,'us_tweets_by_ideology_07-16')
	if not os.path.exists(out_path):
		os.mkdir(out_path)

	with open(tweet_ideology_file,'r') as f:
		ideology_dict = json.load(f)
	all_tweets = parse_tweets.load_all_tweets_from_files(us_tweet_files)

	split_tweets = split_by_ideology(all_tweets,ideology_dict,norm=False)
	write_separate_ideology_files(split_tweets,out_path,suffix='_raw')

	split_tweets_norm = split_by_ideology(all_tweets,ideology_dict,norm=True)
	write_separate_ideology_files(split_tweets_norm,out_path,suffix='_norm')





if __name__ == "__main__":
	main()