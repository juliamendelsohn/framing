from urllib.parse import urlparse
import re
import os
import glob
import json
from parse_tweets import get_all_tweets
import pandas as pd
from collections import Counter
base_path = '/shared/2/projects/framing/data/immigration_tweets_by_country_07-16'
filenames = glob.glob(os.path.join(base_path,'2018','EU.tsv'))
print(filenames)
#tweets = get_all_tweets(filenames,'',filter_retweet=True,filter_lang=True)
all_urls = Counter()
for filename in filenames:
	df = pd.read_csv(filename,sep='\t',header=None)
	df.columns = ['user','id_str','text']
	print(df)
	for text in df['text']:
		URLs = re.findall(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", text)
		for url in URLs:
			print(url)
			domain = urlparse(url).netloc
			all_urls[domain] += 1 



print(all_urls.most_common())
