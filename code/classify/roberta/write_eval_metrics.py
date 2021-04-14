import pandas as pd 
import os


frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
all_dfs = []
for frame_type in frame_types:
	eval_results_file = f'/shared/2/projects/framing/models/classify/{frame_type}/08-31-20_5_epochs_lower_thresh/eval_results.txt'
	with open(eval_results_file,'r') as f:
		df =  pd.DataFrame([x.strip('\n').split(' = ') for x in f.readlines()])
		df.columns = ['metric','value']
		df['value'] = df['value'].astype(float)
		df['frame_type'] = frame_type
		df['value'] = df['value'].round(3)
		all_dfs.append(df)
df = pd.concat(all_dfs)
df = df.pivot(index='frame_type', columns='metric', values='value')
out_file = f'/shared/2/projects/framing/models/classify/08-31-20_5_epochs_lower_thresh_eval.tsv'
df.to_csv(out_file,sep='\t')