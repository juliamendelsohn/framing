import os
import glob
import gzip
import bz2
import json

years = ['2018','2019']
countries = ['EU','GB','US']

for year in years:
	for country in countries:
		data_file = f'/shared/2/projects/framing/data/immigration_tweets_by_country_07-16/{year}/{country}.gz'
		with gzip.open(data_file,'r') as f:
			for i,line in enumerate(f):
				tweet_obj = json.loads(line.decode('utf-8').strip())
				print tweet_obj
				break
					return json.loads(line.decode('utf-8').strip())




# For each country/year 
# Loop through file 
	#For each tweet:
		#Check if the corresponding picture file is in /shared/0
			#If it is, copy to m2/twitter_cache/year/country/{user_id}_224*224.jpg (or whatever)
			#Add user id string to user_id_pick_already_downloaded list
		#Else:
			#Add user id string
	#Dump users_pic_already_downloaded to file /shared/2/projects/framing/m3/user_pic_info/users_with_cached_pics.txt
	#Dump users_no_pic to file /shared/2/projects/framing/m3/user_pic_info/users_no_cached_pics.txt
	# Add list of user strings 