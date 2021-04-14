import pandas as pd 
import os 
import glob



def load_file_list(filenames):
	all_dfs = []
	for filename in filenames:
		df = pd.read_csv(filename,sep='\t')
		all_dfs.append(df)
	return pd.concat(all_dfs,axis=0)

def load_metadata(filenames):
	metadata = load_file_list(filenames)
	metadata['ideology_norm'] = (metadata.ideology - metadata.ideology.mean())/metadata.ideology.std(ddof=0)
	metadata = metadata[['id_str','year','country','ideology','ideology_norm','libcon_raw','libcon_norm']]
	return(metadata)

def combine_data(out_file,binary_labels,metadata,dropna=False):
	df = metadata.merge(binary_labels,left_on='id_str',right_on='id')
	df = df.drop(columns=['labels','text','id'])
	df = df.loc[:, ~df.columns.str.match('Unnamed')]

	if dropna:
		out_file = out_file.split('.')[0] + '_dropna.tsv'
		df = df.dropna()
	df.to_csv(out_file,sep='\t')



def main():
	base_path = '/shared/2/projects/framing/data/labeled_data/'
	binary_label_files = glob.glob(os.path.join(base_path,'dataset_11-03-20','roberta','*','all_frames.tsv'))
	metadata_files = glob.glob(os.path.join(base_path,'dataset_11-03-20','full.tsv'))
	print(metadata_files)
	binary_labels = load_file_list(binary_label_files)
	metadata = load_metadata(metadata_files)
	for dropna in [True,False]:
		out_file = os.path.join(base_path,'stats_table_11-03-20.tsv')
		combine_data(out_file,binary_labels,metadata,dropna=dropna)


if __name__ == "__main__":
	main()