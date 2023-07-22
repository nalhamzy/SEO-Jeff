from pandas.core.frame import DataFrame
from serpapi import GoogleSearch
import os
import csv
import time
import pandas as pd
import numpy as np
from requests import get
import re 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from pprint import pprint
from requests import session
import textrazor
import pandas as pd
import streamlit as st
from keybert import KeyBERT
textrazor.api_key = "cbf491222196a84d4bbcf85f575a75ea323c329bb97a2bb280404dac"

app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'
client = textrazor.TextRazor(extractors=["entities", "topics"])
client.set_cleanup_mode(cleanup_mode='cleanHTML')
client.set_cleanup_return_cleaned(True)
TOP_K_KEYWORDS = 15

from sklearn.feature_extraction.text import TfidfVectorizer

############ tfidf ###############################
def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,TOP_K_KEYWORDS)
    keys = []
    for k in keywords:
        keys.append((k,keywords[k]))

    return keys
    return list(keywords.keys())

def get_tfidf_vectorizer(corpora):
    vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3), smooth_idf=True, use_idf=True)
    vectorizer.fit_transform(corpora)
    feature_names = vectorizer.get_feature_names()
    return vectorizer, feature_names

def get_tfidf_keywords(vectorizer, feature_names, corpora):
    tfidf_df = pd.DataFrame(columns=['text','keywords'])

    for doc in corpora:
        text = doc
        keywords = get_keywords(vectorizer, feature_names, text)
        tfidf_df.loc[len(tfidf_df)] = {'text':text,'keywords':keywords}
            
    return tfidf_df

def get_keywords_summary(entity_df):
    url_df = entity_df['url'].unique()
    text_df =  entity_df['text'].unique()

    vectorizer_2 = TfidfVectorizer(analyzer='word', ngram_range=(2,2),  smooth_idf=True, use_idf=True)
    vectorizer_2.fit_transform(list(text_df))
    feature_names_2 = vectorizer_2.get_feature_names_out()

    vectorizer_3 = TfidfVectorizer(analyzer='word', ngram_range=(3,3),  smooth_idf=True, use_idf=True)
    vectorizer_3.fit_transform(list(text_df))
    feature_names_3 = vectorizer_3.get_feature_names_out()

    keywords_df = pd.DataFrame(columns=['text','keywords'])
    result = []
    corpora = text_df
    for doc in corpora:
    
        keywords = get_keywords(vectorizer_2, feature_names_2, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}
        keywords = get_keywords(vectorizer_3, feature_names_3, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}

        
    keywords_df


    unique_keys = set()
    for i in range(len(keywords_df)):
        for item in keywords_df.iloc[i]['keywords']:
            unique_keys.add(item)

    keywords_summary_df = pd.DataFrame(columns=['keyword','no_documents','average_weight','max'])

    for u_key in unique_keys:
        scores = []
        cnt = 0
        for i in range(len(keywords_df)):
            for keyword in keywords_df.iloc[i]['keywords']:
                if keyword == u_key:
                    cnt = cnt + 1
                    scores.append(keywords_df.iloc[i]['keywords'][u_key])
                    break
        keywords_summary_df.loc[len(keywords_summary_df)] = {'keyword': u_key, 'no_documents':cnt, 'average_weight':np.average(scores),'max': np.max(scores)}
        return keywords_summary_df

def apply_keybert(data_df):
        url_df = data_df.groupby(['url','text'])['url'].unique()
        text_df =  data_df.groupby(['url','text'])['text'].unique()

        kw_extractor = KeyBERT()
        key_df = pd.DataFrame(columns=['url','text','keywords'])
        for j in range(len(url_df)):
            keywords = kw_extractor.extract_keywords(text_df[j], keyphrase_ngram_range=(1,3),stop_words='english',diversity=0.6,top_n=10,use_mmr=True,
            )
            key_df.loc[len(key_df)] = {'url': url_df[j],'text':text_df[j],'keywords':keywords}
        return key_df

def search_keywords(input_text):
    entity_df = pd.DataFrame(columns=['url','entity','entity_type','text'])
    gsearch = GoogleSearch({
        "q": input_text, 
        "location": "Austin,Texas",
        "num" : "10",
        "api_key": app_secret
    })
    result = gsearch.get_dict()

    final_results= []

    df = pd.DataFrame(columns=['link','title','text'])
    count = 0
    for item in result['organic_results']:
        page_url = item['link']
        title=item['title']
        response = client.analyze_url(item['link'])
        
        
        for entity in response.entities():
            if len(entity.freebase_types) > 0:
                entity_df.loc[len(entity_df)] = {'url': page_url,'title':title, 'entity': entity.id, 'entity_type':str(entity.freebase_types[0]),'text':response.cleaned_text }
    
        
    return entity_df


def main():
    user_input = st.text_input('Google Search', 'unlock iphone')

    if  st.button("Search") and len(user_input) > 3 :
        entity_df =  search_keywords(user_input)
        #st.write(mydf.groupby(['url','entity','entity_type'])['entity'].count().reset_index(
        #        name='Count').sort_values(['url','Count'], ascending=False))
        
        #corpora = mydf['text'].unique()
        ### apply tfidf on corpora 
        st.title('Kewords extraction using TF-IDF')
        df_summary = get_keywords_summary(entity_df)
        vectorizer, feature_names = get_tfidf_vectorizer(corpora)
        tfidf_df = get_tfidf_keywords(vectorizer,feature_names,corpora)
        tfidf_df['url'] = mydf['url'].unique()
        st.write(tfidf_df)
        tfidf_csv = tfidf_df[['url','keywords']].to_csv().encode('utf-8')
        st.download_button(
        label="Download data as CSV",
        data=tfidf_csv,
        file_name='tfidf_keywords.csv',
        mime='text/csv',
        )

        ### apply keybert  #########
        st.title('KeyPhrase extraction using BERT')
        key_df = apply_keybert(mydf)
        st.write(key_df)
        key_df_csv = key_df.to_csv().encode('utf-8')
        st.download_button(
        label="Download data as CSV",
        data=key_df_csv,
        file_name='bert_keywords.csv',
        mime='text/csv',
    )

if __name__ == '__main__':
    main()