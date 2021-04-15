import gzip
import csv
import os 
import random
import pandas as pd 
import glob
import json

"""
Sample tweets for human annotation (80% train, 10% dev, 10% test)
Each split contained equal numbers of tweets from all regions (EU, US, GB) and years (2018,2019)
"""

def get_already_annotated(annotated_path):
	files = [os.path.join(annotated_path,x) for x in os.listdir(annotated_path)]
	annotated_ids = []
	for filename in files:
		if filename.split('.')[-1] == 'tsv':
			df = pd.read_csv(filename,sep='\t')
			annotated_ids += [str(s) for s in df['tweet_id']]
		elif filename.split('.')[-1] == 'json':
			with open(filename,'r') as f:
				for line in f:
					tweet = json.loads(line)
					tid = str(tweet['id_str'])
					annotated_ids.append(tid)
	return set(annotated_ids)

def get_samples(filename,train_num,dev_num,test_num,already_annotated_ids):
	all_tweets = []
	with open(filename,'r') as tsvin:
		reader = csv.reader(tsvin,delimiter='\t')
		for row in reader:
			if row[0] not in already_annotated_ids:
				all_tweets.append(row)
				already_annotated_ids.add(row[0])
	num_from_file = train_num + dev_num + test_num
	sampled_tweets = random.sample(all_tweets,num_from_file)
	tweet_sample = {}
	tweet_sample['train'] = sampled_tweets[:train_num]
	tweet_sample['dev'] = sampled_tweets[train_num:train_num + dev_num]
	tweet_sample['test'] = sampled_tweets[-test_num:]
	return tweet_sample


def write_samples(out_file,tweets,id_index,text_index):
	with open(out_file,'w') as f:
		for tweet in tweets:
			entry = {}
			entry['id_str'] = tweet[id_index]
			entry['text'] = tweet[text_index]
			json.dump(entry, f)
			f.write('\n')


def combine_splits(out_path,out_path_sep):
	for split in ['train','dev','test']:
		filenames = glob.glob(os.path.join(out_path_sep,f'*{split}.json'))
		combined_file = os.path.join(out_path,f'{split}.json')
		with open(combined_file,'w') as outfile:
			for fname in filenames:
				with open(fname) as infile:
					for line in infile:
						outfile.write(line)


def main():
	years = [2018,2019]
	countries = ['US','GB','EU']
	data_path = f'/shared/2/projects/framing/data/immigration_tweets_by_country_07-16/'
	out_path = f'/shared/2/projects/framing/data/dataset_07-23/'
	if not os.path.exists(out_path):
		os.mkdir(out_path)

	annotated_path = f'/shared/2/projects/framing/data/sampled_for_annotation/'
	already_annotated_ids = get_already_annotated(annotated_path)

	num_per_file = 750
	train_num = int(0.8*num_per_file)
	dev_num = int(0.1*num_per_file)
	test_num = int(0.1*num_per_file)
	id_index = 1
	text_index=2

	for year in years:
		for country in countries:
			filename = os.path.join(data_path,str(year),country+'.tsv')
			samples = get_samples(filename,train_num,dev_num,test_num,already_annotated_ids)
			for split in samples.keys():
				out_path_sep = os.path.join(out_path,'by_year_country')
				if not os.path.exists(out_path_sep):
					os.mkdir(out_path_sep)
				out_file = os.path.join(out_path_sep,f'{year}-{country}-{split}.json')
				write_samples(out_file,samples[split],id_index,text_index)
	combine_splits(out_path,out_path_sep)



if __name__ == "__main__":
	main()