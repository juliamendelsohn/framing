import json
import gzip
import parse_tweets
from collections import defaultdict,Counter
from preprocess_text import process_text 
import os
import pandas as pd 

#get tweet text
#process text 
#get number of tokens from each lexicon that appear in the text 


def load_lexicons(mfc_pmi_lexicon_file,topn=200):
	pmi_lexicon = defaultdict(list)
	with open(mfc_pmi_lexicon_file,'r') as f:
		pmi_lexicon_full = json.load(f)
	for frame in pmi_lexicon_full:
		if frame not in ['None','Other']:
			words = [x[0] for x in pmi_lexicon_full[frame][:topn]]
			pmi_lexicon[frame] = words
	return pmi_lexicon


def label_tweet(tokens,pmi_lexicon,min_words=2):
	labels = []
	for frame in pmi_lexicon:
		lex = set(pmi_lexicon[frame])
		num_frame_words_in_tweet = len([x for x in tokens if x in lex])
		if num_frame_words_in_tweet >= min_words:
			labels.append(frame)
	return labels

def label_file(filename,pmi_lexicon,min_words=2):
	all_tweets = parse_tweets.load_all_tweets_from_file(filename)
	all_labels = []
	for tweet_obj in all_tweets:
		try:
			tweet_id = tweet_obj['id_str']
			tweet_text = parse_tweets.get_tweet_text(tweet_obj)
			tokens = process_text(tweet_text)
			labels = label_tweet(tokens,pmi_lexicon,min_words=min_words)
			if len(labels) == 0:
				labels = ['None']
			for label in labels:
				result = [tweet_id,label]
				all_labels.append(result)
		except:
			continue
	return all_labels

def label_frames_year_country(tweet_base_path,outfile,pmi_lexicon,start_year,end_year,min_words=2):
	labels_full = []
	for year in range(start_year,end_year+1):
		for country in ['CA', 'GB','EU','US']:
			print(year,country)
			filename = os.path.join(tweet_base_path,str(year),f'{country}.gz')
			labels = label_file(filename,pmi_lexicon,min_words=min_words)
			labels_with_year_country = [[year,country] + l for l in labels]
			labels_full += labels_with_year_country
	labeled_tweets = pd.DataFrame(labels_full)
	labeled_tweets.columns = ['year','country','tid','frame']
	labeled_tweets.set_index('tid')
	labeled_tweets.to_csv(outfile,sep='\t') 


def analyze_frames(filename):
	df = pd.read_csv(filename,sep='\t',index_col=0)
	frame_counts = defaultdict(lambda:Counter())
	frame_counts['full'] = Counter(df['frame'])
	num_tweets = len(set(df['tid']))
	for c in frame_counts['full']:
		frame_counts['full'][c] /= num_tweets
	for country in list(set(df['country'])):
		subset = df[df['country']==country]
		frame_counts[country] = Counter(subset['frame'])
		num_tweets_country = len(set(subset['tid']))
		for c in frame_counts[country]:
			frame_counts[country][c] /= num_tweets_country
	return pd.DataFrame(frame_counts).transpose()



def main():
	min_words = 2
	topn = 200
	start_year = 2011
	end_year = 2020
	mfc_pmi_lexicon_file = "/shared/2/projects/framing/data/pmi_mfc_intersect_thresh_50.json"
	tweet_base_path = "/shared/2/projects/framing/data/immigration_tweets_by_country_04-11/"
	outfile = f"/shared/2/projects/framing/intermediate_results/mfc_labeled_frames_topn{topn}_minwords{min_words}.tsv"
	pmi_lexicon = load_lexicons(mfc_pmi_lexicon_file,topn=topn)
	label_frames_year_country(tweet_base_path,outfile,pmi_lexicon,start_year,end_year,min_words=min_words)
	frame_counts = analyze_frames(outfile)
	print(frame_counts)



if __name__ == "__main__":
	main()