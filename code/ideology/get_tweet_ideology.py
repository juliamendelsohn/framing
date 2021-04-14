import json
import csv
import os 
import parse_tweets
import glob


def load_users_and_ideologies(user_ideology_files):
	user_ideologies = {}
	for filename in user_ideology_files:
		print(filename)
		with open(filename,'r') as f:
			reader = csv.reader(f)
			for row in reader:
				user_id = row[0]
				score = row[1]
				user_ideologies[user_id] = score
	return user_ideologies

def get_tweets_with_ideology(filenames,user_ideologies):
	all_tweets = []
	tweet_ideologies = {}
	for filename in filenames:
		print(filename)
		tweets = parse_tweets.load_all_tweets_from_file(filename)
		for tweet in tweets:
			user_id = tweet['user']['id_str']
			tweet_id = tweet['id_str']
			if user_id in user_ideologies:
				tweet_ideologies[tweet_id] = user_ideologies[user_id]
	return tweet_ideologies

def write_tweet_ideologies(tweet_ideologies,out_path,out_filename):
	outfile = os.path.join(out_path,out_filename)
	with open(outfile,'w') as f:
		json.dump(tweet_ideologies,f)

def get_tweet_files(tweet_base_path):
	filenames = []
	filenames += glob.glob(os.path.join(tweet_base_path,'*','EU.gz'))
	filenames += glob.glob(os.path.join(tweet_base_path,'*','GB.gz'))
	filenames += glob.glob(os.path.join(tweet_base_path,'*','US.gz'))
	return filenames

def main():
	data_dir = '/shared/2/projects/framing/data/'
	tweet_base_path = os.path.join(os.path.join(data_dir,'immigration_tweets_by_country_07-16/'))
	user_ideology_files = glob.glob(os.path.join(data_dir,'tweetscores_ideal_points','*'))

	tweet_files = get_tweet_files(tweet_base_path)
	user_ideologies = load_users_and_ideologies(user_ideology_files)
	tweet_ideologies = get_tweets_with_ideology(tweet_files,user_ideologies)
	write_tweet_ideologies(tweet_ideologies,data_dir,'tweet-ideology-07-16.json')




if __name__ == "__main__":
	main()