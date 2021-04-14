import pandas as pd 
import os



def load_data(models_to_files):
	df_list = []
	for model in models_to_files:
		filename = models_to_files[model]
		df = pd.read_csv(filename,sep='\t').drop(columns='Unnamed: 0')
		df = df.rename(columns={'0': 'Frame Type', '1': 'Frame','2': 'Macro F1'})
		df['Model'] = model
		df_list.append(df)
	return df_list

def combine_dfs(df_list):
	new_df = pd.DataFrame()
	for df in df_list:
		model = df['Model'][0]
		new_df['Frame Type'] = df['Frame Type']
		new_df[f"Frame"] = df['Frame']
		new_df[f"{model}_f1"] = df['Macro F1']
	all_frames_df = new_df[new_df['Frame Type'] == 'all_frames'][['Frame','roberta_finetune_f1']]
	all_frames_df = all_frames_df.rename(columns={'roberta_finetune_f1': 'roberta_finetune_f1_all'})
	new_df = new_df.rename(columns={'roberta_finetune_f1':'roberta_finetune_singlecat'})
	new_df = new_df[new_df['Frame Type'] != 'all_frames']

	result = pd.merge(all_frames_df,new_df,on='Frame',how='outer').sort_values('Frame Type')
	avg_result = result.groupby('Frame Type').agg('mean')
	return result, avg_result





def main():
	models_to_files = {}
	model_dir = '/shared/2/projects/framing/models/classify'
	models_to_files['dummy_random'] = os.path.join(model_dir,'dummy_uniform_macrof1.tsv')
	models_to_files['dummy_most_frequent'] = os.path.join(model_dir,'dummy_most_frequent_macrof1.tsv')
	models_to_files['logreg_unigram'] = os.path.join(model_dir,'logreg_unigram_macrof1.tsv')
	models_to_files['logreg_unigram_bigram'] = os.path.join(model_dir,'logreg_unigram_bigram_macrof1.tsv')
	models_to_files['roberta_finetune'] = os.path.join(model_dir,'09-24-20_eval_by_class_macrof1.tsv')
	date = '09-24-20'

	df_list = load_data(models_to_files)
	df,df_avg = combine_dfs(df_list)
	df_avg.to_csv(os.path.join(model_dir,f'{date}_compare_models_macrof1.tsv'),sep='\t')
	df.to_csv(os.path.join(model_dir,f'{date}_compare_models_by_frame_macrof1.tsv'),sep='\t')





if __name__ == "__main__":
	main()
