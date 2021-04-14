import os
import glob
import pandas as pd
import numpy as np


def get_pred_country_from_frame(path,pattern,countries):
	res = []
	for country in countries:
		filename = os.path.join(path,pattern+country+'.tsv')
		df = pd.read_csv(filename,sep='\t')
		sigdf = df[df['p.value'] < 0.05]
		sigdf['prediction'] = np.where(sigdf['estimate'] > 0 , country, f'Not_{country}')
		print(filename)
		res.append(sigdf)
	df = pd.concat(res)
	print(df)
	outfile = os.path.join(path,'significant_predictors.tsv')
	df.to_csv(outfile,sep='\t')

def get_preds_from_frames(path,pattern):
	res = []
	filenames = glob.glob(os.path.join(path,pattern))
	for filename in filenames:
		df = pd.read_csv(filename,sep='\t')
		frame = os.path.basename(filename).split('_')[0]
		sigdf = df[df['p.value'] < 0.05]
		sigdf = sigdf[sigdf['term'] != '(Intercept)']
		sigdf['prediction'] = np.where(sigdf['estimate'] > 0 , frame, f'Not_{frame}')
		res.append(sigdf)
	df = pd.concat(res)
	df = df.sort_values('term')
	outfile = os.path.join(path,'significant_predictors.tsv')
	#print(df)
	df.to_csv(outfile,sep='\t')



def main():
	base_dir = '/home/juliame/framing/intermediate_results'
	countries = ['US','GB','EU']
	#get_pred_country_from_frame(os.path.join(base_dir,'predict_country_from_frames'),'country_',countries)
	get_preds_from_frames(os.path.join(base_dir,'predict_frames_from_country'), '*_country.tsv')
	get_preds_from_frames(os.path.join(base_dir,'predict_frames_from_ideology'), '*_ideology_norm.tsv')


if __name__ == "__main__":
	main()