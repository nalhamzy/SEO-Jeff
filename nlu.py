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
import numpy as np
app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'
textrazor.api_key = "cbf491222196a84d4bbcf85f575a75ea323c329bb97a2bb280404dac"

client = textrazor.TextRazor(extractors=["entities", "words"])
client.set_cleanup_mode(cleanup_mode='cleanHTML')
client.set_cleanup_return_cleaned(True)



def get_related_terms(text):
    query_related_terms = {}
    params = {
            "q": text,
            "tbm": "isch",
            "ijn": "0",
            "api_key": app_secret
            }

    search = GoogleSearch(params)
    results = search.get_dict()
    query_related_terms[text] = []
    for item in results['suggested_searches']:
                query_related_terms[text].append(item['name']) 
    return query_related_terms

def search_keywords(input_text,num_pages):
    entity_df = pd.DataFrame(columns=['url','entity','entity_type','text','relevanceScore'])
    filter = "inurl:blog|com --intitle:test"

    gsearch = GoogleSearch({
        "q": input_text + ' ' + filter, 
        "location": "Austin,TX,Texas,United States",
        "num" : num_pages,
        "api_key": app_secret
    })
    result = gsearch.get_dict()

    final_results= []

    full_df = pd.DataFrame(columns=['link','title','text'])
    count = 0
    words_pos = {}
    print('lenght of results', len(result['organic_results']))
    for item in result['organic_results']:
        page_url = item['link']
        title=item['title']
        try:
            response = client.analyze_url(item['link'])
            full_df.loc[len(full_df)] = {'link': page_url,'title':title, 'text':response.cleaned_text }
            response_obj = response.json
            for entity in response.entities():
                
                if len(entity.freebase_types) > 0:
                    ## check if entity doesnt exist in entity_df
                    # if entity_df[entity_df['entity'] == str(entity.id).lower()   ].empty: 

                    entity_df.loc[len(entity_df)] = {'url': page_url,'title':title, 
                    'entity': str(entity.id).lower(), 'entity_type':str(entity.freebase_types[0]), 
                    'text':response.cleaned_text, 'relevanceScore':entity.json['relevanceScore'] }

            
            if 'response' in response_obj and 'sentences' in response_obj['response']:
                for sent in response_obj['response']['sentences']:
                    for word in sent['words']:
                        words_pos[word['token']] = word['partOfSpeech']
    
                
        except Exception as ex:
            print('exception caught in search_keywords', ex)
    entity_df = entity_df.drop_duplicates(subset=['entity', 'url'])
    return entity_df, words_pos, full_df