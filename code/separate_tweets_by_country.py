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
from collections import defaultdict


def load_user_country_info(user_country_filename):
	with gzip.open(user_country_filename,'r') as f:
		json_bytes = f.read()
	json_str = json_bytes.decode('utf-8')
	user_dict = json.loads(json_str)
	return user_dict

def load_eu_list(eu_country_list_file):
	with open(eu_country_list_file,'r') as f:
		eu_countries = set([x.strip('\n') for x in f.readlines()])
	return eu_countries

def separate_tweets_by_country(all_tweets,user_dict,eu_countries):
	all_tweets_shuffled = random.sample(all_tweets,len(all_tweets)) 
	tweets_by_country = defaultdict(list)
	completed_tweets = set()
	for tweet_obj in all_tweets_shuffled:
		try:
			if tweet_obj['id_str'] not in completed_tweets:
				user_id_str = tweet_obj['user']['id_str']
				if user_id_str in user_dict:
					country = user_dict[user_id_str]
					if country in eu_countries:
						tweets_by_country['EU'].append(tweet_obj)
					else:
						tweets_by_country[country].append(tweet_obj)
		except:
			continue
		completed_tweets.add(tweet_obj['id_str'])
	return tweets_by_country


def write_tweets_by_country(outdir,tweets_by_country):
	for country in tweets_by_country:
		outfile_gz = os.path.join(outdir,f'{country}.gz')
		outfile_tsv = os.path.join(outdir,f'{country}.tsv')
		tweets = tweets_by_country[country]
		parse_tweets.write_tweets(outfile_gz,tweets)
		parse_tweets.write_tweets_text(outfile_tsv,tweets)


def separate_by_country_by_year(start_year,end_year,data_path_base,data_pattern_base,outdir_base,user_dict,eu_countries,query):
	for year in range(start_year,end_year+1):
		data_path = os.path.join(data_path_base,str(year))
		pattern = f'{data_pattern_base}_{year}*.gz'
		filenames = glob.glob(os.path.join(data_path,pattern))
		outdir = os.path.join(outdir_base,str(year))
		if not os.path.exists(outdir):
			os.mkdir(outdir)
		all_tweets = parse_tweets.get_all_tweets(filenames,query,filter_retweet=True,filter_lang=True,filter_query=True)
		tweets_by_country = separate_tweets_by_country(all_tweets,user_dict,eu_countries)
		write_tweets_by_country(outdir,tweets_by_country)


def main():

	user_country_filename = "/shared/1/projects/framing/data/user_countries.json.gz"
	eu_country_list_file = "/home/juliame/framing/eu_countries.txt"
	data_path_base = f'/shared/2/projects/framing/data/immigration_tweets/'
	data_pattern_base = '*'
	outdir_base = f'/shared/2/projects/framing/data/immigration_tweets_by_country_07-16/'
	if not os.path.exists(outdir_base):
		os.mkdir(outdir_base)

	user_dict = load_user_country_info(user_country_filename)
	eu_countries = load_eu_list(eu_country_list_file)
	query = "immigration|immigrants?|illegals|undocumented|illegal aliens?|migrants?|migration"

	separate_by_country_by_year(2018,2019,data_path_base,data_pattern_base,outdir_base,user_dict,eu_countries,query)


	


if __name__ == "__main__":
	main()