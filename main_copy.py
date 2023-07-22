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
textrazor.api_key = "cbf491222196a84d4bbcf85f575a75ea323c329bb97a2bb280404dac"
from pattern.en import lexeme
from nltk.corpus import stopwords
import spacy
import voice_type 
import keyword_summarizer as ks
import nlu as ee
import unit_extractor as ue
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import sent_tokenize
units_df = pd.read_csv('unit.csv')
try: 
    lexeme('dog')
except: 
    pass


s_words = set(stopwords.words('english'))

app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'
client = textrazor.TextRazor(extractors=["entities", "words"])
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
    return keywords
def get_tfidf_vectorizer(corpora):
    vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3), smooth_idf=True, use_idf=True)
    vectorizer.fit_transform(corpora)
    feature_names = vectorizer.get_feature_names()
    return vectorizer, feature_names


def get_numeric_values(text_batch):
    numeric_pairs = {}
    numeric_df_op2 = pd.DataFrame(columns=['Numeric Value','Sentence'])
    numeric_df = pd.DataFrame(columns=['Value','Unit', 'Key','Doc','Sentence','No_Documents'])
    doc_num = 0
    import difflib
    units_df['unit']  = units_df['unit'].astype(str)
    units_df['unit'] = units_df['unit'].apply(lambda x: x.lower())
    


    for text in text_batch:
        doc_num += 1
        sentences = sent_tokenize(text)

        for sent in sentences:
            
            result = nlp(sent)
            for token_idx in range(len(result)):
                token = result[token_idx]
                if token.tag_ == "CD":
                    if token_idx + 1 < len(result) :
                        next_token = result[token_idx + 1]
                        if next_token.tag_ == "NN" or next_token.tag_ == "NNS" or next_token.tag_ == "SYM":
                            if next_token.text != '-':
                                key = token.text.lower() + " " + next_token.text.lower()
                                match = difflib.get_close_matches(next_token.text.lower(), units_df['unit'].values, cutoff=0.95)
                                if len(match) > 0:

                                    numeric_df.loc[len(numeric_df)] = {'Value':token.text, 'Unit':next_token.text ,'Key': key, 'Doc':doc_num, 'Sentence':sent}
                                    print("match found {}- token.text {} , next_token.text {}".format(match, token.text, next_token.text))

                                    if key not in numeric_pairs:
                                            
                                            numeric_pairs[key] =  set([doc_num])
                                    else:
                                            curr_set = numeric_pairs[key]
                                            curr_set.add(doc_num)
                    elif token_idx - 1 >= 0:
                        prev_token = result[token_idx - 1]
                        key = token.text.lower() + " " + prev_token.text.lower()
                        if key not in numeric_pairs:
                                
                                numeric_pairs[key] =  set([doc_num])
                        else:
                                print("unit found in diff document")
                                curr_set = numeric_pairs[key]
                                curr_set.add(doc_num)
                        if prev_token.tag_ == "NN" or prev_token.tag_ == "NNS" or prev_token.tag_ == "SYM":
                            if prev_token.text != "-":
                                match = difflib.get_close_matches(prev_token.text.lower(), units_df['unit'].values, cutoff=0.95)
                                if len(match) > 0:
                                    print("match found {}- prev_token.text {} , token.text {}".format(match, prev_token.text, token.text))
                                    numeric_df.loc[len(numeric_df)] = {'Value':token.text, 'Unit':prev_token.text,'Key':key ,'Doc':doc_num, 'Sentence':sent}
    
    print("Numeric Pairs Dict")
    print(numeric_pairs)
    print("unique numeric df")
    unique_numeric_df = numeric_df[numeric_df.duplicated(['Key','Doc']) == False]
    unique_numeric_df = unique_numeric_df[unique_numeric_df.duplicated(['Key'])]
    
    
    for idx, row in unique_numeric_df.iterrows():
        unique_numeric_df.at[idx,'No_Documents'] = int(len(numeric_pairs[row.Key]))
    print('after loop')

    print(unique_numeric_df)

    return unique_numeric_df

    print("Numeric Pairs Dict")
    print(numeric_pairs)
    for key in numeric_pairs:
        if len(numeric_pairs[key]) > 1:
            print("key {}".format(key))
            print(numeric_pairs[key])
            print(numeric_df[numeric_df.Key == key]['Sentence'])
            numeric_df_op2.loc[len(numeric_df_op2)] = {'Numeric Value': key, 'Sentence':numeric_df[numeric_df.Key == key]['Sentence']}
    
    print('df')
    print(numeric_df)

    print('df2')
    print(numeric_df_op2)
    return numeric_df, numeric_df_op2

def get_keywords_summary(entity_df,threshold, words_pos):
    url_df = entity_df['url'].unique()
    text_df =  entity_df['text'].unique()

    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1,1),  smooth_idf=True, use_idf=True)
    vectorizer.fit_transform(list(text_df))
    feature_names = vectorizer.get_feature_names_out()

    vectorizer_2 = TfidfVectorizer(analyzer='word',stop_words='english',  ngram_range=(2,2),  smooth_idf=True, use_idf=True)
    vectorizer_2.fit_transform(list(text_df))
    feature_names_2 = vectorizer_2.get_feature_names_out()

    vectorizer_3 = TfidfVectorizer(analyzer='word',stop_words='english',  ngram_range=(3,3),  smooth_idf=True, use_idf=True)
    vectorizer_3.fit_transform(list(text_df))
    feature_names_3 = vectorizer_3.get_feature_names_out()

    keywords_df = pd.DataFrame(columns=['text','keywords'])
    result = []
    corpora = text_df
    for doc in corpora:
        keywords = get_keywords(vectorizer, feature_names, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}

        keywords = get_keywords(vectorizer_2, feature_names_2, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}
        keywords = get_keywords(vectorizer_3, feature_names_3, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}

        


    unique_keys = set()
    for i in range(len(keywords_df)):
        for item in keywords_df.iloc[i]['keywords']:
            unique_keys.add(item)

    keywords_summary_df = pd.DataFrame(columns=['keyword','no_documents','computed_score','average_weight','max','pos','is_entity','ngram','query_suffixes_prefixes'])

    for u_key in unique_keys:
        scores = []
        cnt = 0
        for i in range(len(keywords_df)):
            for keyword in keywords_df.iloc[i]['keywords']:
                if keyword == u_key:
                    cnt = cnt + 1
                    scores.append(keywords_df.iloc[i]['keywords'][u_key])
                    break
        computed_score = np.average(scores) * np.power(cnt,2)
        if computed_score > threshold:
            pos = ''
            query_suffixes_prefixes = []
            if u_key in words_pos:
                pos = words_pos[u_key]
                query_suffixes_prefixes = lexeme(u_key)
            is_entity = False 
            if u_key in entity_df['entity'].unique():
                is_entity = True
            keywords_summary_df.loc[len(keywords_summary_df)] = {'keyword': u_key, 'no_documents':cnt,'computed_score':computed_score, 
                                                                 'average_weight':np.average(scores),'max': np.max(scores),'pos': pos,
                                                                 'is_entity':is_entity, 'ngram': len(u_key.split(' ')),'query_suffixes_prefixes':query_suffixes_prefixes }
    return keywords_summary_df



def search_keywords(input_text,num_pages):
    entity_df = pd.DataFrame(columns=['url','entity','entity_type','text'])
    gsearch = GoogleSearch({
        "q": input_text, 
        "location": "Austin,TX,Texas,United States",
        "num" : num_pages,
        "api_key": app_secret
    })
    result = gsearch.get_dict()

    final_results= []

    df = pd.DataFrame(columns=['link','title','text'])
    count = 0
    words_pos = {}
    for item in result['organic_results']:
        page_url = item['link']
        title=item['title']
        try:
            response = client.analyze_url(item['link'])
            response_obj = response.json
            
            for entity in response.entities():
                if len(entity.freebase_types) > 0:
                    entity_df.loc[len(entity_df)] = {'url': page_url,'title':title, 
                    'entity': str(entity.id).lower(), 'entity_type':str(entity.freebase_types[0]), 
                    'text':response.cleaned_text }

            
            if 'response' in response_obj and 'sentences' in response_obj['response']:
                for sent in response_obj['response']['sentences']:
                    for word in sent['words']:
                        words_pos[word['token']] = word['partOfSpeech']
                        
        except Exception:
            print('exception caught')
    return entity_df, words_pos


def main():
    threshold = st.slider('Score threshold (avg_weight * number_of_doc^2):',0.0, 1.0,0.2,0.05)
    no_pages = st.selectbox(
    'Number of pages:',
    ('10', '20', '30','40','50','60','70','80','90','100','110','120','140','150'))
    user_input = st.text_input('Google Search', 'unlock iphone')

    if  st.button("Search",no_pages) and len(user_input) > 3 :
        entity_df,words_pos =  ee.search_keywords(user_input,no_pages)
        passive_df = voice_type.process_df(entity_df)
        #st.write(mydf.groupby(['url','entity','entity_type'])['entity'].count().reset_index(
        #        name='Count').sort_values(['url','Count'], ascending=False))

        #corpora = mydf['text'].unique()
        ### apply tfidf on corpora 
        keywords_summary_df = ks.get_keywords_summary(entity_df, threshold, words_pos)
        st.title('Kewords extraction using TF-IDF')
        tfidf_csv = keywords_summary_df.sort_values('computed_score',ascending=False).to_csv().encode('utf-8')  
        st.write(keywords_summary_df.sort_values('computed_score',ascending=False).head())        
        st.download_button(
        label="Download full data as CSV",
        data=tfidf_csv,
        file_name='tfidf_keywords.csv',
        mime='text/csv',
        )

        numeric_df = ue.get_numeric_values(entity_df['text'].unique())

        st.write("Numeric Values")
        st.dataframe(numeric_df[['Key','Sentence','No_Documents']])
        #if len(numeric_df) > 0:
            #st.write('---- Numeric Values that appeared more than once (in one or more document) --- ')
 
            #st.dataframe(unique_numeric_df[unique_numeric_df.Count > 1])
        
        #st.write('---- Numeric Values that appeared in more than one document (url) --- ')
        #st.dataframe(numeric_df_op2)
        
        
        suffixes_prefixes = {}
        for item in user_input.split():
            if item not in s_words:
                suffixes_prefixes[item] = lexeme(item)


     
        if len(suffixes_prefixes) > 0:
            st.write("Suffixes & Prefixes")
            st.write(suffixes_prefixes)


        query_related_terms = {}
        params = {
            "q": user_input,
            "tbm": "isch",
            "ijn": "0",
            "api_key": app_secret
            }

        search = GoogleSearch(params)
        results = search.get_dict()
        query_related_terms[user_input] = []
        for item in results['suggested_searches']:
                query_related_terms[user_input].append(item['name'])    
        
        if len(query_related_terms) > 0:
            st.write("Query Related Terms")
            st.write(query_related_terms)
     
        if len(passive_df) > 0:
            st.title('Passive Voice Detection')
            st.write(passive_df[['url','is_passive']])

if __name__ == '__main__':
    main()