from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm
import re
from os import path
import pandas as pd

'''
    module de transformation des données textuelles
'''

def tfidfVectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.5,
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def remove_punct(text_list):
    mod_text=[]
    for text in tqdm(text_list, desc="removing punct") :
        text = re.sub(r'\W', ' ', text).strip()
        mod_text.append(text)
    return mod_text

def lemmatize(lang, mode, text_list, punct='wthtPunct'):
    filename='data/deft09_parlement_'+mode+'_'+lang+'_data_lemmatized'+punct+'.lst'
    if path.exists(filename):
        with open(filename, 'r') as f:
            new_text_data = f.read().splitlines()
    else:
        if lang in ['fr', 'it']:
            model = lang +'_core_news_sm'
        else :
            model = 'en_core_web_sm'
        nlp = spacy.load(model)
        sw_spacy = nlp.Defaults.stop_words
        new_text_data = []
        with open(filename, 'w') as f:
            for texts in tqdm(text_list, desc="lemmatizing"):
                doc = nlp(texts)
                words=[]
                for token in doc :
                    word = token.lemma_
                    if word.lower() not in sw_spacy and re.match('\w+', word) and len(word) > 2 :
                        words.append(word.lower())
                new_text = " ".join(words)
                new_text_data.append(new_text)
                f.write(str(new_text)+'\n')

    return new_text_data

def balance_data(data, target, size=None):
    annot_text = {
        "data":data,
        "target":target
    }
    df = pd.DataFrame(annot_text)
    eldr = df[df.target=='eldr']
    gue_ngl = df[df.target=='gue-ngl']
    ppe_de = df[df.target=='ppe-de']
    pse = df[df.target=='pse']
    verts_ale = df[df.target=='verts-ale']
    if size == None:
        size = min([len(eldr), len(gue_ngl), len(ppe_de), len(pse), len(verts_ale)])

    eldr_resampled = resample(eldr, replace=True, n_samples=size, random_state=123)
    gue_ngl_resampled = resample(gue_ngl, replace=True, n_samples=size, random_state=123)
    ppe_de_resampled = resample(ppe_de, replace=True, n_samples=size, random_state=123)
    pse_resampled = resample(pse, replace=True, n_samples=size, random_state=123)
    verts_ale_resampled = resample(verts_ale, replace=True, n_samples=size, random_state=123)

    df_downsampled = pd.concat([eldr_resampled, gue_ngl_resampled, ppe_de_resampled, pse_resampled, verts_ale_resampled])
    return df_downsampled['data'].tolist(), df_downsampled['target'].tolist()