import parse_tweets
import os
import glob
import gzip
import bz2
import json
import re
from itertools import product
from multiprocessing import Pool


def get_files(dates):
	filenames = []
	for date in dates:
		year = str(int(date.split('-')[0]))
		month = str(int(date.split('-')[1]))

		if int(year) < 2018:
			if year in ['2011','2012']:
				filename = f'/euler-twitter/storage3/gardenhose/raw/{year}/{month}/gardenhose.{date}.gz'
			elif year == '2013':
				filename = f'/euler-twitter/storage4/gardenhose/raw/{year}/{month}/gardenhose.{date}.gz'
			elif year == '2014':
				filename = f'/euler-twitter/storage5/gardenhose/raw/{year}/{month}/gardenhose.{date}.gz'
			elif year in ['2015','2016','2017']:
				filename = f'/euler-twitter/storage2/gardenhose/raw/{year}/{month}/gardenhose.{date}.gz'
			if os.path.exists(filename):
				filenames.append(filename)

		else:
			files = glob.glob(f'/twitter-turbo/decahose/raw/decahose.{date}.*')
			filenames += files
			if len(files) == 0:
				strdate = ''.join(date.split('-'))
				strfiles = glob.glob(f'/twitter-turbo/decahose/raw/decahose.*{strdate}*.*')
				for f in strfiles:
					if f not in filenames:
						filenames.append(f)
	return filenames

def get_tweets_from_file(filename,outdir,outfile_prefix,query,antiquery=None):
	compression_type = filename.split('.')[-1] #either gz or bz2
	ending = '.'.join(os.path.basename(filename).split('.')[1:-1])
	year = ending[:4]

	outdir_year = os.path.join(outdir,year)
	outfile = os.path.join(outdir_year,f'{outfile_prefix}_{ending}.gz')
	all_tweets = []

	if not os.path.exists(outdir_year):
		os.mkdir(outdir_year)

	if not os.path.exists(filename):
		print(f"Input file does not exist: {filename}")
		return

	if os.path.exists(outfile):
		print(f"Output file already exists: {outfile}")
		return 
	try:
		if compression_type == 'gz':
			f = gzip.open(filename)
		else:
			f = bz2.open(filename)
		with f:
			print(f"Opening file: {filename}")
			for i,line in enumerate(f):
				if i%100000 == 0:
					print(i,ending,len(all_tweets))
				query_found = parse_tweets.search_line(line,query)
				if query_found != None:
					if antiquery != None and parse_tweets.search_line(line,antiquery) == None:
						all_tweets.append(query_found)
	except:
		print(f"Error in processing file: {filename}")

	parse_tweets.write_tweets(outfile,all_tweets)


def main():
	years = [str(i) for i in range(2018,2020)]
	months = [str(i).zfill(2) for i in range(1,13)] #1 to 13 (12)
	days = [str(i).zfill(2) for i in range(1,32)] #1 to 32 (31)
	dates = [y + '-' + m+'-'+d for (y,m,d) in product(years,months,days)]

	outdir = '/shared/2/projects/framing/data/immigration_tweets/'
	outfile_prefix = "missing_tweets"
	#query = "immigration|immigrants?|illegals|undocumented|illegal aliens"
	query = "illegal alien|migrants?|migration"
	antiquery = "immigration|immigrants?|illegals|undocumented|illegal aliens"


	filenames = get_files(dates)
	func_tuples = [(f,outdir,outfile_prefix,query,antiquery) for f in filenames]
	pool = Pool(12)
	pool.starmap(get_tweets_from_file,func_tuples)

if __name__ == "__main__":
	main()