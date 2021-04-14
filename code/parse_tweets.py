import os 
import gzip
import bz2
import json
import csv
import re
import glob
from itertools import product
import random 
from preprocess_text import process_text

#assume that files containing tweets are .gz
def get_all_tweets(filenames,query,filter_retweet=True,filter_lang=True,filter_query=True):
	all_tweets = []
	for filename in filenames:
		print(filename)
		with gzip.open(filename,'r') as f:
			for i,line in enumerate(f):
				tweet_obj = load_tweet_obj(line)
				tweet_text = get_tweet_text(tweet_obj)
				if filter_tweets(tweet_obj,tweet_text,query,filter_retweet,filter_lang,filter_query):
					all_tweets.append(tweet_obj)
	return all_tweets

#this one doesn't filter
def load_all_tweets_from_files(filenames):
	all_tweets = []
	for filename in filenames:
		with gzip.open(filename,'r') as f:
			for i,line in enumerate(f):
				tweet_obj = load_tweet_obj(line)
				all_tweets.append(tweet_obj)
	return all_tweets


def load_tweet_obj(line):
	return json.loads(line.decode('utf-8').strip())

def convert_tweet_object_to_string(tweet_obj):
	obj_str = json.dumps(tweet_obj) + '\n'
	obj_bytes = obj_str.encode('utf-8')
	return obj_bytes 

def get_tweet_text(obj):
	if 'text' not in obj and 'extended_tweet' not in obj:
		return None
	if 'extended_tweet' in obj:
		tweet_text = obj['extended_tweet']['full_text']
	else:
		tweet_text = obj['text']
	return tweet_text.replace('\t',' ').replace('\n',' ') 


#returns false if tweet does not pass filter, true if it does
#filter_query makes sure that the query is found in the text after text processing (tokenization, removing usernames, etc)
def filter_tweets(tweet_obj,tweet_text,query=None,filter_retweet=True,filter_lang=False,filter_query=False): #WHen true, filters out tweets that don't comply
	if filter_retweet:
		if ('retweeted_status' in tweet_obj) or (tweet_text[:2] == 'RT'):
			return False
	if filter_lang:
		if ('lang' in tweet_obj and tweet_obj['lang']!='en'):
			return False
	if filter_query:
		assert(query != None)
		token_string = ' '.join(process_text(tweet_text))
		if (re.search(query,token_string,re.IGNORECASE) == None):
			return False
	return True

def search_line(line,query):
	tweet_obj = load_tweet_obj(line)
	tweet_text = get_tweet_text(tweet_obj)
	if tweet_text != None:
		if filter_tweets(tweet_obj,tweet_text) and re.search(query,tweet_text,re.IGNORECASE) != None:
			return tweet_obj
	return None

def write_tweets(outfile,tweets):
	with gzip.open(outfile,'w') as f:
		for tweet_obj in tweets:
			tweet_string = convert_tweet_object_to_string(tweet_obj)
			f.write(tweet_string)

def write_tweets_text(outfile,tweets):
	with open(outfile,'w') as tsvout:
		writer = csv.writer(tsvout,delimiter='\t')
		for tweet in tweets:
			try:
				tweet_text = get_tweet_text(tweet)
				username = tweet['user']['screen_name']
				id_str = tweet['id_str']
				writer.writerow([username,id_str,tweet_text])
			except:
				continue



#get all usernames from a gz file containing tweet objects
#max_following to restrict to users following fewer than max_following others
def get_usernames(filename,max_following=None):  
	users = []
	with gzip.open(filename) as f:
		for line in enumerate(f):
			obj = load_tweet_obj(line)
			username = obj['user']['screen_name'] 
			if (max_following is None) or (int(obj['user']['friends_count']) <= max_following):
				users.append(username)
	return users

def get_all_usernames(tweet_path,tweet_pattern):
	all_users = set()
	filelist = glob.glob(os.path.join(tweet_path,tweet_pattern))
	for filename in filelist:
		print(filename)
		users = get_usernames(filename)
		all_users = all_users | users
	return list(all_users) 

def write_usernames(users,outfile,shuffle=True):
	if shuffle:
		random.shuffle(users)
	with open(outfile,'w') as f:
		for user in userlist:
			f.write(user + '\n')














