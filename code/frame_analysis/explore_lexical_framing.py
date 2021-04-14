import csv 
import re
from collections import defaultdict,Counter
import os
import json
import nltk
from preprocess_text import process_text
from nltk import word_tokenize
from nltk.util import ngrams
import parse_tweets
import glob


def formulate_regex_query(nouns):
	full_query = ''
	for word in nouns:
		query = r'\w+\s+' + word + r'|'
		full_query += query 
	full_query = full_query[:-1]
	return full_query


def get_word_counts(filenames,lex_frame_query):
	all_tweets = parse_tweets.load_all_tweets_from_files(filenames)
	word_count = defaultdict(list)
	for i,tweet in enumerate(all_tweets):
		if i % 1000 == 0:
			print(i)
		text = parse_tweets.get_tweet_text(tweet)
		tokens= process_text(text)
		word_count['lex_frame_bigrams'] += [x.lower() for x in re.findall(lex_frame_query,text,re.IGNORECASE)]
		word_count['unigrams'] += tokens
		word_count['bigrams'] += list(zip(*[tokens[i:] for i in range(2)]))
	return word_count


def write_word_counts(out_path,word_counts):
	for ngram_type in word_counts:
		out_file = os.path.join(out_path,ngram_type + '.json')
		wc_dict = Counter(word_counts[ngram_type])
		if type(word_counts[ngram_type][0]) == tuple:
			wc_dict = {  ' '.join(k):v for (k,v) in wc_dict.items()}
		with open(out_file,'w') as f:
			json.dump(wc_dict,f)

def combine_word_counts(out_path,out_filename,filenames_to_combine):
	out_file = os.path.join(out_path,out_filename)
	full_counts = Counter()
	for filename in filenames_to_combine:
		with open(filename,'r') as f:
			file_counts = json.load(f)
			full_counts += Counter(file_counts)
	with open(out_file,'w') as f2:
		json.dump(full_counts,f2)

def get_word_counts_by_country(tweet_base_path,word_count_base_path,countries,lex_frame_query):
	country_path = os.path.join(word_count_base_path,'by_country')
	if not os.path.exists(country_path):
		os.mkdir(country_path)

	for country in countries:
		filenames = glob.glob(os.path.join(tweet_base_path,'*',country + '.gz'))
		print(filenames)
		word_count = get_word_counts(filenames,lex_frame_query)
		out_path = os.path.join(country_path,country)
		if not os.path.exists(out_path):
			os.mkdir(out_path)

		write_word_counts(out_path,word_count)



def get_word_counts_by_ideology(tweet_base_path,word_count_base_path,ideologies,lex_frame_query):
	ideology_path = os.path.join(word_count_base_path,'by_ideology')
	if not os.path.exists(ideology_path):
		os.mkdir(ideology_path)
	for i in ideologies:
		filename = os.path.join(tweet_base_path,i + '.gz')
		word_count = get_word_counts([filename],lex_frame_query)
		out_path = os.path.join(ideology_path,i)
		if not os.path.exists(out_path):
			os.mkdir(out_path)
		write_word_counts(out_path,word_count)



def combine_all_ngram_counts(word_count_path,out_path,ngram_types):
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	for ngram_type in ngram_types:
		files_to_combine = glob.glob(os.path.join(word_count_path,'*',ngram_type+'.json'))
		combine_word_counts(out_path,ngram_type+'.json',files_to_combine)







def main():
	nouns = ['immigrants?',
			'immigration',
			'emigration',
			'emigrants?',
			'migration',
			'migrants?']
	lex_frame_query = formulate_regex_query(nouns)
	data_dir = '/shared/2/projects/framing/data/'
	tweet_base_path = os.path.join(data_dir,'immigration_tweets_by_country_07-16/')
	tweet_ideology_path = os.path.join(data_dir,'us_tweets_by_ideology_07-16')
	word_count_base_path = '/shared/2/projects/framing/intermediate_results/word_counts/'

	if not os.path.exists(word_count_base_path):
		os.mkdir(word_count_base_path)

	get_word_counts_by_country(tweet_base_path,word_count_base_path,['EU'],lex_frame_query)
	# ideologies = ['conservative_norm','liberal_norm','conservative_raw','liberal_raw']
	# get_word_counts_by_ideology(tweet_ideology_path,word_count_base_path,ideologies,lex_frame_query)

	ngram_types = ['lex_frame_bigrams','bigrams','unigrams']
	country_path = os.path.join(word_count_base_path,'by_country')
	combined_country_out_path = os.path.join(country_path,'full_counts')
	combine_all_ngram_counts(country_path,combined_country_out_path,ngram_types)









if __name__ == '__main__':
	main()