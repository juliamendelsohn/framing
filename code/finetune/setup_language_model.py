from preprocess_text import tokenize
from parse_tweets import get_tweet_text, load_all_tweets_from_files
import glob
import gzip
import json
import random
import os
import re


"""
What this script needs to do. 
Iterate through all tweets in dataset. 
Decide if it goes to train, dev, or test (random number 0-7 go to train, 8 to dev , 9 to test)


Append it as a line in either immigration_tweets.train.raw immigration_tweets.dev.raw or immigration_tweets.test.raw
Replace tabs and newlines with spaces 
"""


def get_all_tokens(filenames,all_lower=False,remove_hashtag=False):
	tweets = load_all_tweets_from_files(filenames)
	token_strings = []
	for tweet in tweets:
		text = get_tweet_text(tweet)
		token_string = tokenize(text,all_lower=all_lower,remove_hashtag=remove_hashtag)
		token_strings.append(token_string)
	return token_strings


def separate_token_strings(token_strings,train_dev_test_split=(80,10,10)):
	assert (len(train_dev_test_split) == 3)
	assert (sum(train_dev_test_split) == 100)

	shuffled_strings = random.sample(token_strings,len(token_strings))
	num_train = int((train_dev_test_split[0]/100) * len(shuffled_strings))
	num_dev = int((train_dev_test_split[1]/100) * len(shuffled_strings))
	num_test = int((train_dev_test_split[2]/100) * len(shuffled_strings))

	train = shuffled_strings[0:num_train]
	dev = shuffled_strings[num_train:(num_train+num_dev)]
	test = shuffled_strings[(num_train+num_dev):]

	return train,dev,test


def write_to_file(out_path,out_filename,text,keep_case=True):
	if not os.path.exists(out_path):
		os.mkdir(out_path)

	with open(os.path.join(out_path,out_filename),'w') as f:
		for token_string in text:
			if keep_case:
				f.write(token_string + '\n')
			else:
				f.write(token_string.lower() + '\n')

def write_all_splits_to_file(out_path,train,dev,test,keep_case=True):
	write_to_file(out_path,'train.raw',train,keep_case=keep_case)
	write_to_file(out_path,'dev.raw',dev,keep_case=keep_case)
	write_to_file(out_path,'test.raw',test,keep_case=keep_case)


def main():

	data_path = f'/shared/2/projects/framing/data/immigration_tweets_by_country_07-16'
	data_patterns = [os.path.join(data_path,'*',c + '.gz') for c in ['EU','GB','US']]
	filenames = []
	for pattern in data_patterns:
		filenames += glob.glob(pattern)

	out_path_base = f'/shared/2/projects/framing/data/lm_data_08-25/'
	if not os.path.exists(out_path_base):
		os.mkdir(out_path_base)

	cased_path = os.path.join(out_path_base,'cased')
	uncased_path = os.path.join(out_path_base,'uncased')
	
	token_strings = get_all_tokens(filenames)
	train,dev,test = separate_token_strings(token_strings)
	write_all_splits_to_file(cased_path,train,dev,test,keep_case=True)
	write_all_splits_to_file(uncased_path,train,dev,test,keep_case=False)



if __name__ == "__main__":
	main()