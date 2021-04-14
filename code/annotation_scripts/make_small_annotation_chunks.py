import gzip
import csv
import os 
import random
import pandas as pd 
import json

def get_files(samples_path,year,countries):
	files = []
	for country in countries:
		filename = os.path.join(samples_path,f'{year}-{country}.tsv')
		files.append(filename)
	return files


# load first 20 tweets from each file 
# shuffle first 20 tweets from each file
# write 60 tweets to out file
# just shuffle from outfile text --> we can match up later using the tweet id user
def write_samples(files,out_file,start_line,end_line):
	chunk = []
	for filename in files:
		with open(filename,'r') as tsvin:
			reader = csv.reader(tsvin,delimiter='\t')
			for i,row in enumerate(reader):
				if i >= start_line and i <= end_line:
					chunk.append(row)
	sample_to_annotate = random.sample(chunk,len(chunk))
	with open(out_file,'w') as f:
		writer = csv.writer(f,delimiter='\t')
		writer.writerow(['tweet_id','text'])
		for entry in sample_to_annotate:
			writer.writerow(entry)



def write_samples_potato(files,out_file,num_from_file,already_annotated_ids):
	tweet_sample = []
	for filename in files:
		tweets = []
		with open(filename,'r') as tsvin:
			reader = csv.reader(tsvin,delimiter='\t')
			for row in reader:
				tweets.append(row)
		not_yet_annotated = [t for t in tweets if t[0] not in already_annotated_ids]
		tweets_to_annotate = random.sample(not_yet_annotated,num_from_file)
		tweet_sample += tweets_to_annotate

	shuffled_tweet_sample = random.sample(tweet_sample,len(tweet_sample))
	with open(out_file,'w') as f:
		for tweet in shuffled_tweet_sample:
			entry = {}
			entry['id_str'] = tweet[0]
			entry['text'] = tweet[1]
			json.dump(entry, f)
			f.write('\n')

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


def main():
	year = 2018
	countries = ['US','GB','EU']
	samples_path = f'/shared/1/projects/framing/data/sampled_immigration_tweets/'
	files = get_files(samples_path,year,countries)
	out_path = f'/shared/1/projects/framing/data/sampled_for_annotation/'
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	date = '05-07-2020-2'
	out_file = os.path.join(out_path,date +'.json')

	#2-05-20: lines 0-19
	#2-15-20: lines 20-39
	#3-9-20: lines 40-59
	#3-25-20: lines 60-79
	#4-2-20: lines 80-99
	#4-17-20: lines 100-119
	#start_line = 100
	#end_line = 119
	#write_samples(files,out_file,start_line,end_line)
	already_annotated_ids = get_already_annotated(out_path)
	num_from_file = 10
	write_samples_potato(files,out_file,num_from_file,already_annotated_ids)






if __name__ == "__main__":
	main()