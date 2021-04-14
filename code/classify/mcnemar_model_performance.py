from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd 
import numpy as np 
from itertools import combinations
import os






def make_contigency_table(model1_results,model2_results):
	df1 = pd.read_csv(model1_results,sep='\t').drop(columns=['Unnamed: 0'])
	df2 = pd.read_csv(model2_results,sep='\t').drop(columns=['Unnamed: 0'])
	both_correct = 0
	both_wrong = 0
	first_correct = 0
	second_correct = 0
	for frame in df1.columns:
		for i,elem in enumerate(df1[frame]): 
			first = df1[frame][i]
			second = df2[frame][i]
			if first == True and second == True:
				both_correct += 1
			elif first == True and second == False:
				first_correct += 1
			elif first == False and second == True:
				second_correct += 1
			else:
				both_wrong += 1
	table = np.array([[both_correct,first_correct],
			[second_correct,both_wrong]
			])
	
	return table

def make_contigency_table_full_correctness(model1_results,model2_results):
	df1 = pd.read_csv(model1_results,sep='\t').drop(columns=['Unnamed: 0']).all(axis=1)
	df2 = pd.read_csv(model2_results,sep='\t').drop(columns=['Unnamed: 0']).all(axis=1)
	both_correct = 0
	both_wrong = 0
	first_correct = 0
	second_correct = 0

	for i in range(len(df1)):
		first = df1[i]
		second = df2[i]
		if first == True and second == True:
			both_correct += 1
		elif first == True and second == False:
			first_correct += 1
		elif first == False and second == True:
			second_correct += 1
		else:
			both_wrong += 1

	table = np.array([[both_correct,first_correct],
			[second_correct,both_wrong]
			])
	
	return table






def main():
	base_path = '/shared/2/projects/framing/models/classify/11-14-20_test/'
	frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
	model_names = ['dummy_random','dummy_stratified', 'dummy_frequent',
	'logreg_unigram','logreg_bigram','roberta_baseline','roberta_finetune']
	seeds = [35, 12, 45, 42, 23]
	mcnemar_results = []
	for frame_type in frame_types:
		for (model1,model2) in combinations(model_names,2):
			table = np.array([[0,0],[0,0]])
			for seed in seeds:
				model1_file = os.path.join(base_path,frame_type,model1,f'seed_{seed}_by_sample.tsv')
				model2_file = os.path.join(base_path,frame_type,model2,f'seed_{seed}_by_sample.tsv')
				t = make_contigency_table(model1_file,model2_file)
				#t = make_contigency_table_full_correctness(model1_file,model2_file)
				table += t
			test = mcnemar(table)
			statistic = test.statistic
			pvalue = test.pvalue
			result = (model1,model2,frame_type,statistic,pvalue,pvalue < 0.00001)
			mcnemar_results.append(result)
			print(result)

	df = pd.DataFrame(mcnemar_results)
	df.columns = ['Model 1','Model 2','Frame Type','Statistic','P-value','p < 0.00001']
	df.to_csv(os.path.join(base_path,'mcnemar.tsv'),sep='\t')



if __name__ == "__main__":
	main()


