import re
import os
import glob
import parse_tweets
import json


lexical_frames = [
'illegal immigrants?',
'illegal immigration',
'mass immigration',
'mass migration',
'immigration',
'immigrants?',
'emigration',
'emigrants?',
'migration',
'migrants?',
'illegals',
'illegal aliens?',
'undocumented']

def formulate_regex_query(lexical_frames):
	full_query = ''
	for frame in lexical_frames:
		#query = r'\b' + frame + r'\b|'
		query = frame + r'|'
		full_query += query 
	full_query = full_query[:-1]
	return full_query


def identify_lexical_frame(filenames,query):
	tweet_lexical_frames = {}
	all_tweets = parse_tweets.get_all_tweets(filenames,query,filter_retweet=False,filter_lang=False,filter_query=False)
	for tweet_obj in all_tweets:
		tweet_text = parse_tweets.get_tweet_text(tweet_obj)
		tweet_id = tweet_obj['id_str']
		frames = list(set([x.lower() for x in re.findall(query,tweet_text,re.IGNORECASE)]))
		tweet_lexical_frames[tweet_id] = frames
	return tweet_lexical_frames

def write_lexical_frames(tweet_lexical_frames,out_path,out_file):
	out_filename = os.path.join(out_path,out_file)
	with open(out_filename,'w') as f:
		json.dump(tweet_lexical_frames,f)

			


def get_tweet_files(tweet_base_path):
	filenames = []
	filenames += glob.glob(os.path.join(tweet_base_path,'*','EU.gz'))
	filenames += glob.glob(os.path.join(tweet_base_path,'*','GB.gz'))
	filenames += glob.glob(os.path.join(tweet_base_path,'*','US.gz'))
	return filenames


def main():
	data_dir = '/shared/2/projects/framing/data/'
	tweet_base_path = os.path.join(data_dir,'immigration_tweets_by_country_07-16/')
	out_path = '/shared/2/projects/framing/intermediate_results/frame_labels/'
	out_file = 'lexical_frames_08-04.json'

	filenames = get_tweet_files(tweet_base_path)
	query = formulate_regex_query(lexical_frames)
	tweet_lexical_frames = identify_lexical_frame(filenames,query)
	write_lexical_frames(tweet_lexical_frames,out_path,out_file)
	






if __name__ == "__main__":
	main()

