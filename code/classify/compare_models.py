import pandas as pd 
import os
from ast import literal_eval




def load_df(base_path,frame_type,model_name,seed):
	result_file = os.path.join(base_path,frame_type,model_name,f'seed_{seed}.tsv')
	df = pd.read_csv(result_file,sep='\t')#,converters={'Support Pearson': literal_eval})
	df['Frame'] = df['Unnamed: 0']
	#df['Correlation'] = [x[0] for x in df['Support Pearson']]
	#df = df.drop(columns=['Unnamed: 0','Support Pearson'])
	df = df.drop(columns=['Unnamed: 0'])
	df = df[df['support'] >= 2] #### Don't account for no support or just one when looking at subsets
	df = df[~df.Frame.str.contains("avg")] ## Remove already calculated averages
	df.set_index(['Frame'], inplace= True)
	df['Seed'] = seed
	df['Frame Type'] = frame_type
	df['Model'] = model_name
	df = df.reset_index()
	return df

def average_performance(df):
	new_df = df[~df.Frame.str.contains("avg")]
	new_df = new_df.groupby(['Seed','Frame Type','Model']).agg('mean').reset_index()
	longdf = pd.melt(new_df,id_vars=['Frame Type','Model','Seed'], value_vars=None, var_name='Metric', value_name='Score')
	return longdf

def compare_model_metric(base_path,frame_type,model1_name,model2_name,seeds,metric):
	model1_dfs = load_all_seeds(base_path,frame_type,model1_name,seeds)
	model2_dfs = load_all_seeds(base_path,frame_type,model2_name,seeds)


def main():
	for eval_set in ['eval_liberal','eval_conservative','eval_US','eval_GB','eval_EU']:
		base_path = f'/shared/2/projects/framing/models/classify/3-30-21_{eval_set}/'
		frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative']
		model_names = ['dummy_random','dummy_stratified','dummy_frequent',
		'logreg_unigram','logreg_bigram','roberta_finetune','roberta_baseline']
		seeds = [35, 12, 45, 42, 23]
		all_dfs = []
		for model_name in model_names:
			for frame_type in frame_types:
				for seed in seeds:
					df = load_df(base_path,frame_type,model_name,seed)
					all_dfs.append(df)

		df = pd.concat(all_dfs,axis=0)
		df.to_csv(os.path.join(base_path,'eval_metrics.tsv'),sep='\t')
		df_eval_metrics = average_performance(df)
		df_eval_metrics.to_csv(os.path.join(base_path,'avg_eval_metrics.tsv'),sep='\t')
			


if __name__ == "__main__":
	main()



