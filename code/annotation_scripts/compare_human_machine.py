import pandas as pd 
import glob
import json
import os
import numpy as np
from collections import Counter
import nltk
from nltk.metrics import masi_distance, agreement
from functools import reduce
from get_final_agreement import *


#eval_set = dev, test, or both
def get_eval_ids(eval_dir,eval_set='both'):
    dev_file = os.path.join(eval_dir,'dev.tsv')
    test_file = os.path.join(eval_dir,'test.tsv')
    dev_ids = list(pd.read_csv(dev_file,sep='\t',dtype=str)['id_str'])
    test_ids = list(pd.read_csv(test_file,sep='\t',dtype=str)['id_str'])
    if eval_set == 'both':
        return set(dev_ids + test_ids)
    elif eval_set == 'dev':
        return set(dev_ids)
    elif eval_set == 'test':
        return set(test_ids)
    
def get_machine_labels(predicted_frame_file,eval_ids,out_file):
    frames = {}
    frames['Issue-General'] = set(['Capacity and Resources', 'Crime and Punishment','Cultural Identity', 
    'Economic','External Regulation and Reputation', 'Fairness and Equality','Health and Safety',
    'Legality, Constitutionality, Jurisdiction', 'Morality and Ethics','Policy Prescription and Evaluation',
    'Political Factors and Implications', 'Public Sentiment','Quality of Life', 'Security and Defense'])
    frames['Issue-Specific'] = set(['Hero: Cultural Diversity','Hero: Integration', 'Hero: Worker',
    'Threat: Fiscal', 'Threat: Jobs', 'Threat: National Cohesion','Threat: Public Order',
    'Victim: Discrimination','Victim: Global Economy', 'Victim: Humanitarian', 'Victim: War'])
    frames['Narrative'] = set(['Episodic','Thematic'])

    predicted_frames = pd.read_csv(predicted_frame_file,dtype=str,sep='\t')
    predicted_frames_eval = predicted_frames[predicted_frames['id_str'].isin(eval_ids)]

    all_entries = []
    with open(out_file,'w') as f:
        for index, row in predicted_frames_eval.iterrows():
            new_entry = {}
            new_entry['id'] = row['id_str']
            new_entry['annotation'] = {}
            for frame_type in frames:
                new_entry['annotation'][frame_type] = []
                for frame in frames[frame_type]:
                    if row[frame] == '1':
                        new_entry['annotation'][frame_type].append(frame)
            f.write(json.dumps(new_entry) + '\n')

def get_annot_df_from_file(filename,frame_types):
    annots = load_annotations(filename,frame_types)
    df = pd.melt(annots,id_vars='id_str')
    df.columns = ['id_str','frame_type','frames']
    return df

def get_human_annots(annotation_path,frame_types,aggregate = False):
    human_annot_files = glob.glob(os.path.join(annotation_path,'*','*_all.jsonl'))
    human_annot_files = sorted([f for f in human_annot_files if "consensus_all.jsonl" not in f])
    coder_dfs = [load_annotations(f,frame_types) for f in human_annot_files]
    human1_dfs = [df for (i,df) in enumerate(coder_dfs) if i % 2 == 0]
    human2_dfs = [df for (i,df) in enumerate(coder_dfs) if i % 2 == 1]
    
    df1 = pd.melt(pd.concat(human1_dfs),id_vars='id_str')
    df2 = pd.melt(pd.concat(human2_dfs),id_vars='id_str')
    df1.columns = ['id_str','frame_type','frames']
    df2.columns = ['id_str','frame_type','frames']

    if aggregate:
        return pd.concat([df1,df2])
    else:
        return (df1,df2)

def calculate_human_consensus_agreement(annotation_path,consensus_file,frame_types):
    consensus_df = get_annot_df_from_file(consensus_file)
    human_df = get_human_annots(annotation_path,frame_types,aggregate = True)
    agreement_df = calculate_agreement(consensus_df,machine_df,frame_types,'consensus','human')
    return agreement_df


def calculate_machine_consensus_agreement(machine_labels_file,consensus_file,frame_types):
    consensus_df = get_annot_df_from_file(consensus_file)
    machine_df = get_annot_df_from_file(machine_labels_file)
    agreement_df = calculate_agreement(consensus_df,machine_df,frame_types,'consensus','machine')
    return agreement_df

def calculate_agreement(df1,df2,frame_types,name1,name2):
    combined_df = pd.merge(df1,df2,on=['id_str','frame_type'],how='right')
    combined_df.columns = ['id_str','frame_type',f'labels-{name1}',f'labels-{name2}']
    agreement_df = overall_agreement(combined_df,(name1,name2),frame_types)
    return agreement_df

def compare_human_agreement_with_machine(annotation_path,machine_labels_file,frame_types):
    machine_df = get_annot_df_from_file(machine_labels_file,frame_types)
    (human1_df,human2_df) = get_human_annots(annotation_path,frame_types,aggregate = False)
    agree_h_m = calculate_agreement(pd.concat([human1_df,human2_df]),machine_df,frame_types,'human','machine')
    agree_h1_h2 =  calculate_agreement(human1_df,human2_df,frame_types,'human1','human2')
    agreement_df = pd.concat([agree_h_m,agree_h1_h2])
    return agreement_df



def write_results(out_file,human_df,machine_df):
    human_df['Annotator'] = 'Human'
    machine_df['Annotator'] = 'Machine'
    df = pd.concat([human_df,machine_df])
    df = df.drop(columns=['Coder'])
    df.to_csv(out_file,sep='\t')
    


def main():
    human_annotation_path = '/home/juliame/framing/labeled_data/annotations_by_pair'
    consensus_file = '/home/juliame/framing/labeled_data/eval_annots.jsonl'
    eval_dir = '/shared/2/projects/framing/data/labeled_data/dataset_11-03-20'
    predicted_frame_file = '/shared/2/projects/framing/data/full_datasheet_11-13-20.tsv'
    machine_labels_for_agreement_file = '/home/juliame/framing/labeled_data/eval_machine_predictions.jsonl'
    out_file = '/home/juliame/framing/labeled_data/eval_agreement_betw_humans_and_machine.tsv'
    frame_types = ['Issue-General','Issue-Specific','Narrative']

    #eval_ids = get_eval_ids(eval_dir)
    #get_machine_labels(predicted_frame_file,eval_ids,machine_labels_for_agreement_file)
    # human_consensus_agreement = calculate_human_consensus_agreement(annotation_output_path_base,consensus_file,frame_types)
    # machine_consensus_agreement = calculate_machine_consensus_agreement(machine_labels_for_agreement_file,consensus_file,frame_types)
    # write_results(out_file,human_consensus_agreement,machine_consensus_agreement)
    #get_human_annots(annotation_output_path_base,frame_types)
    agreement_df = compare_human_agreement_with_machine(human_annotation_path,machine_labels_for_agreement_file,frame_types)
    print(agreement_df)
    agreement_df.to_csv(out_file,sep='\t')

if __name__ == "__main__":
    main()


