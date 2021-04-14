import pandas as pd 
import os
import json 
import glob 

def add_lexical_frames(existing_df,lexical_frame_file):
	with open(lexical_frame_file,'r') as f:
		lex_frames = json.load(f)

	df = existing_df.copy()
	df['lex'] = [lex_frames[str(id_str)] for id_str in existing_df['id_str']]
	df['lex'] = df['lex'].replace('immigrants','immigrant')
	df['lex'] = df['lex'].replace('migrants','migrant')
	df['lex'] = df['lex'].replace('emigrants','emigrant')
	df['lex'] = df['lex'].replace('illegal aliens','illegal alien')
	df['lex'] = df['lex'].replace('illegal immigrants','illegal immigrant')

	df = pd.concat([df,pd.get_dummies(df['lex'].apply(pd.Series).stack()).sum(level=0)],axis=1)
	df = df.drop(columns=['Unnamed: 0','lex'])
	return df


def add_follow_info(df,follow_file):
	df1 = pd.read_csv(follow_file)[['id','num_followers','num_followed']]
	df = df.merge(df1,left_on='id_str',right_on='id')
	return df

def add_response_info(df,response_dir):
	response_dfs = []
	for filename in os.listdir(response_dir):
		response_df = pd.read_csv(os.path.join(response_dir,filename),sep='\t',dtype=str)
		response_dfs.append(response_df)
	response = pd.concat(response_dfs,axis=0)
	response = response.drop(columns=['Unnamed: 0'])
	df['id_str']=df['id_str'].astype(str)
	df = df.merge(response,left_on='id_str',right_on='id',how='left')
	return df


def add_other_metadata_features(df,metadata_file):
	metadata = pd.read_csv(metadata_file,sep='\t',dtype=str)
	df = df.merge(metadata,on='id_str',how='left')
	return df




def main():
	stats_table_dir = '/home/juliame/framing/labeled_data/'
	data_dir = '/shared/2/projects/framing/data/'
	lexical_frame_file = '/shared/2/projects/framing/intermediate_results/frame_labels/lexical_frames_08-04.json'
	metadata_file = '/shared/2/projects/framing/data/train_metadata_features_10-21.tsv'

	follow_file = os.path.join(data_dir,'tweet_metadata_07-21.csv')
	response_dir = os.path.join(data_dir,'num_fav_rt_07-16')

	old_stats_file = 'stats_table_09-04-20.tsv'
	expanded_stats_file = 'stats_table_10-21-20.tsv'

	existing_df = pd.read_csv(os.path.join(stats_table_dir,old_stats_file),sep='\t')
	#df = add_lexical_frames(existing_df,lexical_frame_file)
	df = add_follow_info(existing_df,follow_file)
	df = add_response_info(df,response_dir)
	df = add_other_metadata_features(df,metadata_file)
	df = df.drop(columns=['id_x','id_y','user'])
	df.to_csv(os.path.join(stats_table_dir,expanded_stats_file),sep='\t')


if __name__ == "__main__":
	main()