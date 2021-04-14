import pandas as pd 
import glob
import json
import os
import numpy as np
from collections import Counter

# 2. Model performance by ideology and country
# - Get mapping from user id to country/ideology from /shared/2/projects/framing/data/labeled_data/dataset_11-03-20
# - Create new dev and test sets for EU, UK, US, US-Lib, US-Cons, splits from /shared/2/projects/framing/data/labeled_data/dataset_11-03-20/roberta/ 
# - Run evaluation scripts 

def separate_data(eval_dir,eval_set):
    eval_file = os.path.join(eval_dir,f'{eval_set}.tsv')
    df = pd.read_csv(eval_file,sep='\t',dtype=str)
    for country in ['EU','GB','US']:
        df_subset = df[df['country']==country]
        out_file = os.path.join(eval_dir,f'{eval_set}_{country}.tsv')
        df_subset.to_csv(out_file,sep='\t')
    for ideology in ['liberal','conservative']:
        df_subset = df[(df['country']=='US') & (df['libcon_raw']==ideology)]
        out_file = os.path.join(eval_dir,f'{eval_set}_{ideology}.tsv')
        df_subset.to_csv(out_file,sep='\t')
    


def main():
    eval_dir = '/shared/2/projects/framing/data/labeled_data/dataset_11-03-20'
    separate_data(eval_dir,eval_set='dev')
    separate_data(eval_dir,eval_set='test')


if __name__ == "__main__":
    main()