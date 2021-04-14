import re
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score
from sklearn.dummy import DummyClassifier



def load_data(train_path,eval_path,frame_type):
    train_file = os.path.join(train_path,frame_type + '.tsv')
    eval_file = os.path.join(eval_path,frame_type + '.tsv')
    df_train = pd.read_csv(train_file,sep='\t')
    df_eval = pd.read_csv(eval_file,sep='\t')
    frames = [x for x in df_train.columns if x not in ['Unnamed: 0', 'text', 'id','labels']]
    all_data = defaultdict(lambda: defaultdict(list))
    for frame in frames:
        all_data[frame]['X_train'] = df_train.text
        all_data[frame]['X_eval'] = df_eval.text
        all_data[frame]['y_train'] = df_train[frame]
        all_data[frame]['y_eval'] = df_eval[frame]
    return all_data



def run_classifier(data,frame,strategy):
    X_train = data[frame]['X_train']
    X_eval = data[frame]['X_eval']
    y_train = data[frame]['y_train']
    y_eval = data[frame]['y_eval']
    clf = DummyClassifier(strategy=strategy,random_state=42)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_eval)
    f1 = f1_score(y_eval,y_pred,average='macro')
    return f1



def main():
    frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
    base_path = '/shared/2/projects/framing/data/labeled_data/splits_08-16-20/roberta/'
    out_path = '/shared/2/projects/framing/models/classify/'

    for strategy in ['most_frequent','uniform']:
        out_file = os.path.join(out_path,f'dummy_{strategy}_macrof1.tsv')
        all_results = []
        for frame_type in frame_types:
            train_path = os.path.join(base_path,'train')
            eval_path = os.path.join(base_path,'dev')
            data = load_data(train_path,eval_path,frame_type)
            for frame in data.keys():
                f1 = run_classifier(data,frame,strategy)
                results = [frame_type,frame,f1]
                all_results.append(results)
        df = pd.DataFrame(all_results,columns = ['Frame Type','Frame','Macro F1'])
        df.to_csv(out_file,sep='\t')




if __name__ == "__main__":
    main()

        




