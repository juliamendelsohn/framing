import os 
import random
import json
import gzip
import glob
import csv
from preprocess_twitter import tokenize
import re


eu_country_list_file = "/home/juliame/framing/eu_countries.txt"
with open(eu_country_list_file,'r') as f:
	eu_countries = set([x.strip('\n') for x in f.readlines()])

def load_user_country_info(filename):
	with gzip.open(filename,'r') as f:
		json_bytes = f.read()
	json_str = json_bytes.decode('utf-8')
	user_dict = json.loads(json_str)
	return user_dict



def get_all_tweets(data_path,pattern):
	all_files = glob.glob(os.path.join(data_path,pattern))
	all_tweets = []
	for filename in all_files:
		print(filename)
		with gzip.open(filename,'r') as f:
			for i,line in enumerate(f):
				try:
					obj = json.loads(line.decode('utf-8').strip())
					#if ('lang' in obj and obj['lang'] == 'en') or obj['user']['lang']=='en':
					if ('lang' in obj and obj['lang']=='en'):

						if 'extended_tweet' in obj:
							tweet_text = obj['extended_tweet']['full_text']
						else:
							tweet_text = obj['text']

						token_string = tokenize(tweet_text,keep_hashtag=True)
						query = "immigration|immigrants?|illegals|undocumented|illegal aliens"
						if (re.search(query,token_string,re.IGNORECASE) != None):
							obj_str = json.dumps(obj) + '\n'
							obj_bytes = obj_str.encode('utf-8')
							all_tweets.append(obj_bytes)
				except:
					continue
	return all_tweets

def sample_tweets(all_tweets,num,outpath,outpattern):
	if not os.path.exists(outpath):
		os.mkdir(outpath)
	sample_tweets = random.sample(all_tweets,num)
	outfile_tweet = os.path.join(outpath,outpattern + '.gz')
	outfile_text = os.path.join(outpath,outpattern + '.tsv')
	with gzip.open(outfile_tweet,'w') as f:
			for tweet in sample_tweets:
				f.write(tweet)
	with open(outfile_text,'w') as tsvout:
		writer = csv.writer(tsvout,delimiter='\t')
		for tweet in sample_tweets:
			obj = json.loads(tweet)
			tid = obj['id_str']
			text = obj['text'].replace('\t',' ').replace('\n',' ')
			writer.writerow([tid,text])


def sample_tweets_by_country(all_tweets,user_dict,country_code,num):  # country_code either US, GB, CA, EU. for EU we need to instead check if country is in eu_countries
	all_shuffled_tweets = random.sample(all_tweets,len(all_tweets))
	sampled_tweets = []
	for tweet in all_shuffled_tweets:
		obj = json.loads(tweet)
		user_id_str = obj['user']['id_str']
		if user_id_str in user_dict:
			if country_code == 'EU' and (user_dict[user_id_str] in eu_countries):
				sampled_tweets.append(tweet)
			elif user_dict[user_id_str] == country_code:
				sampled_tweets.append(tweet)
		if (num != None) and (len(sampled_tweets) >= num):
			break
	return sampled_tweets

def write_sampled_tweets(sampled_tweets,out_path,out_pattern):
	outfile_tweet = os.path.join(out_path,out_pattern + '.gz')
	outfile_text = os.path.join(out_path,out_pattern + '.tsv')

	with gzip.open(outfile_tweet,'w') as f:
		for tweet in sampled_tweets:
			f.write(tweet)
	with open(outfile_text,'w') as tsvout:
		writer = csv.writer(tsvout,delimiter='\t')
		for tweet in sampled_tweets:
			obj = json.loads(tweet)
			tid = obj['id_str']
			if 'extended_tweet' in obj:
				tweet_text = obj['extended_tweet']['full_text']
			else:
				tweet_text = obj['text']
			text = tweet_text.replace('\t',' ').replace('\n',' ')
			writer.writerow([tid,text])



def main():

	for year in range(2011,2021):
		#num = 1000
		num = None
		user_country_filename = "/shared/1/projects/framing/data/user_countries.json.gz"
		data_path = f'/shared/1/projects/framing/data/immigration_tweets/{year}/'
		pattern = f'immigrant_tweets_{year}*.gz'
		all_tweets = get_all_tweets(data_path,pattern)
		user_dict = load_user_country_info(user_country_filename)

		out_path = f'/shared/1/projects/framing/data/immigration_tweets_by_country/{year}'
		if not os.path.exists(out_path):
			os.mkdir(out_path)



		for country_code in ['EU','US','GB','CA']:
			out_pattern = f'{year}-{country_code}'
			sampled_tweets = sample_tweets_by_country(all_tweets,user_dict,country_code,num)
			write_sampled_tweets(sampled_tweets,out_path,out_pattern)


	

if __name__ == "__main__":
	main()