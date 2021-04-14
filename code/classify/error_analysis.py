import pandas as pd
import os
import csv





def load_annotations(annotation_file):
	df = pd.read_csv(annotation_file,sep='\t').drop(columns=['Unnamed: 0'])
	return df


def load_predictions(prediction_file):
	df = pd.read_csv(prediction_file,sep='\t').drop(columns=['Unnamed: 0'])
	return df


def find_errors(df_annot,df_predict):
	df_errors_list = []
	for frame in df_predict.columns:
		error_indexes = df_predict.index[df_predict[frame] == False].tolist()
		df_annot_errors = df_annot.iloc[error_indexes][['id','text',frame]]
		df_annot_errors['Mistake'] = [f"erroneously predicted {frame}" if x == 0 else f"missed {frame}" for x in df_annot_errors[frame]]
		df_annot_errors = df_annot_errors.drop(columns=frame)
		df_errors_list.append(df_annot_errors)
	return pd.concat(df_errors_list,axis=0)




def main():
	for frame_type in ['Issue-General','Issue-Specific','Narrative']:
		annotation_file = f'/shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta/test/{frame_type}.tsv'
		prediction_file = f'/shared/2/projects/framing/models/classify/11-14-20_test/{frame_type}/roberta_finetune/seed_12_by_sample.tsv'
		out_file = f'/shared/2/projects/framing/results/{frame_type}_error_analysis.tsv'
		df_annot = load_annotations(annotation_file)
		df_predict = load_predictions(prediction_file)
		errors = find_errors(df_annot,df_predict)
		errors.to_csv(out_file,sep='\t')
		print(errors)



if __name__ == "__main__":
	main()