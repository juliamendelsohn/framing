import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


def load_data(train_path,eval_path,frame_type):
    all_data = defaultdict(list)
    train_file = os.path.join(train_path,frame_type + '.tsv')
    df_train = pd.read_csv(train_file,sep='\t')

    eval_file = os.path.join(eval_path,frame_type + '.tsv')
    df_eval = pd.read_csv(eval_file,sep='\t')

    all_data['X_train'] = df_train.text
    all_data['X_eval'] = df_eval.text

    all_data['y_train'] = df_train['labels']
    all_data['y_eval'] = df_eval['labels']
    return all_data



def run_classifier(data,seed,include_bigrams=True):
    vec = CountVectorizer(ngram_range=(1, 1), lowercase=True)
    if include_bigrams:
        vec = CountVectorizer(ngram_range=(1, 2), lowercase=True)
    X_train = vec.fit_transform(data['X_train'])
    X_eval = vec.transform(data['X_eval'])

    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(), random_state=seed)
    clf.fit(X_train,data['y_train'])

    y_pred = clf.predict(X_eval)
    support_train = data['y_train'].sum()
    support_eval = data['y_eval'].sum()
    f1 = f1_score(data['y_eval'],y_pred,average='macro')
    all_f1 = f1_score(data['y_eval'],y_pred,average=None) 
    lrap = label_ranking_average_precision_score(data['y_eval'],y_pred)
    print(f1)
    print(all_f1)
    return f1, all_f1, lrap, support_train, support_eval



def main():
    frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
    base_path = '/shared/2/projects/framing/data/labeled_data/splits_08-16-20/roberta/'
    out_path = '/shared/2/projects/framing/models/classify/'

    include_bigrams = False
    # if include_bigrams:
    #     out_file = os.path.join(out_path,'logreg_unigram_bigram_macrof1.tsv')
    # else:
    #     out_file = os.path.join(out_path,'logreg_unigram_macrof1.tsv')
    # all_results = []
    for frame_type in frame_types[:1]:
        train_path = os.path.join(base_path,'train')
        eval_path = os.path.join(base_path,'dev')
        data = load_data(train_path,eval_path,frame_type)
        seed = 35
        f1, all_f1, lrap, support_train, support_eval = run_classifier(data,seed,include_bigrams=include_bigrams)
        results = [frame_type,f1,lrap,support_train,support_eval]
        print(results)
        #all_results.append(results)
   
    # print(all_results)
    #df = pd.DataFrame(all_results,columns = ['Frame Type','Macro F1','Train Support','Dev Support','Top Features'])
    #df.to_csv(out_file,sep='\t')


if __name__ == "__main__":
    main()
