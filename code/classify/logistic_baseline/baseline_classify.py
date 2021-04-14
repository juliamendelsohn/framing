import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import f1_score
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# from nltk.stem.snowball import SnowballStemmer
import os
from collections import defaultdict
from sklearn.metrics import classification_report

# stemmer = SnowballStemmer("english")
# def stemmed_words(doc):
#     return (stemmer.stem(w) for w in analyzer(doc))
# analyzer = CountVectorizer(ngram_range=(1,1)).build_analyzer()

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



def run_classifier(data,frame,include_bigrams=True):
    vec = CountVectorizer(ngram_range=(1, 1), lowercase=True)
    if include_bigrams:
        vec = CountVectorizer(ngram_range=(1, 2), lowercase=True)
    X_train = vec.fit_transform(data[frame]['X_train'])
    X_eval = vec.transform(data[frame]['X_eval'])
    clf = LogisticRegression()
    clf.fit(X_train,data[frame]['y_train'])
    y_pred = clf.predict(X_eval)
    support_train = data[frame]['y_train'].sum()
    support_eval = data[frame]['y_eval'].sum()
    f1 = f1_score(data[frame]['y_eval'],y_pred,average='macro')

    coefs = clf.coef_[0]
    feature_names = vec.get_feature_names()
    features = list(zip(feature_names,coefs))
    top_features = [feat[0] for feat in sorted(features, key=lambda x: x[1],reverse=True)][:10]
    return f1, support_train, support_eval, top_features


#         coefs = LogReg_pipeline.steps[1][1].coef_[0]
#         feature_names = LogReg_pipeline.steps[0][1].get_feature_names()
#         features = list(zip(feature_names,coefs))
#         top_features = [feat[0] for feat in sorted(features, key=lambda x: x[1],reverse=True)][:10]
#         bottom_features = [feat[0] for feat in sorted(features, key=lambda x: x[1])][:10]
#         top_list.append(top_features)
#         bottom_list.append(bottom_features)


def main():
    frame_types = ['Issue-General','Issue-Specific','Issue-Specific-Combined','Narrative','all_frames']
    base_path = '/shared/2/projects/framing/data/labeled_data/splits_08-16-20/roberta/'
    out_path = '/shared/2/projects/framing/models/classify/'

    include_bigrams = True
    if include_bigrams:
        out_file = os.path.join(out_path,'logreg_unigram_bigram_macrof1.tsv')
    else:
        out_file = os.path.join(out_path,'logreg_unigram_macrof1.tsv')
    all_results = []
    for frame_type in frame_types:
        train_path = os.path.join(base_path,'train')
        eval_path = os.path.join(base_path,'dev')
        data = load_data(train_path,eval_path,frame_type)
        for frame in data.keys():
            results = [frame_type,frame] + list(run_classifier(data,frame,include_bigrams=include_bigrams))
            all_results.append(results)
    print(all_results)
    df = pd.DataFrame(all_results,columns = ['Frame Type','Frame','Macro F1','Train Support','Dev Support','Top Features'])
    df.to_csv(out_file,sep='\t')




if __name__ == "__main__":
    main()

        




