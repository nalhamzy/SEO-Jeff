from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from pattern.en import lexeme

try: 
    lexeme('dog')
except: 
    pass

TOP_K_KEYWORDS = 15
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

