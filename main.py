from pandas.core.frame import DataFrame
from serpapi import GoogleSearch

import pandas as pd
from requests import get
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from pprint import pprint
from requests import session
import textrazor
import pandas as pd
import streamlit as st
textrazor.api_key = "cbf491222196a84d4bbcf85f575a75ea323c329bb97a2bb280404dac"

import pandas as pd
import numpy as np
from requests import get
import re 
# nltk.download('punkt')
# nltk.download('stopwords')
from pprint import pprint
from requests import session
import textrazor
import pandas as pd
import numpy as np
google_app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'
textrazor.api_key = "cbf491222196a84d4bbcf85f575a75ea323c329bb97a2bb280404dac"

client = textrazor.TextRazor(extractors=["words","phrases","entities"])
# client.set_cleanup_mode(cleanup_mode='cleanHTML')
# client.set_cleanup_return_cleaned(True)


# from nltk.corpus import stopwords
# import spacy
# import voice_type 
# import keyword_summarizer as ks
# import nlu as nlu
# import unit_extractor as ue
# nlp = spacy.load("en_core_web_sm")
# from nltk.tokenize import sent_tokenize



# s_words = set(stopwords.words('english'))
def search_keywords(input_text,num_pages=100):
    entity_df = pd.DataFrame(columns=['url','entity','entity_type','text','relevanceScore'])
    filter = "inurl:,edu OR inurl:.gov -intitle:page -inurl:page"
    print( "'" + input_text + "'" + ' ' + filter)
    gsearch = GoogleSearch({
        "q": '"' + input_text + '"' + ' ' + filter,
        "location": "United States",
        "num" : num_pages,
        "api_key": google_app_secret
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
            print("Analyze: " + item['title'])
            response = client.analyze(item['title'])
            full_df.loc[len(full_df)] = {'link': page_url,'title':title, 'text':title }
            response_obj = response.json
            for entity in response.entities():
                if len(entity.freebase_types) > 0:
                    ## check if entity doesnt exist in entity_df
                    # if entity_df[entity_df['entity'] == str(entity.id).lower()   ].empty: 
                    print('found')
                    entity_df.loc[len(entity_df)] = {'url': page_url,'title':title, 
                    'entity': str(entity.id).lower(), 'entity_type':str(entity.freebase_types[0]), 
                    'text':title, 'relevanceScore':entity.json['relevanceScore'] }

            
            if 'response' in response_obj and 'sentences' in response_obj['response']:
                for sent in response_obj['response']['sentences']:
                    for word in sent['words']:
                        words_pos[word['token']] = word['partOfSpeech']
    
                
        except Exception as ex:
            print('exception caught in search_keywords', ex)
    entity_df = entity_df.drop_duplicates(subset=['entity', 'url'])
    return entity_df, words_pos, full_df


def main():
    # threshold = st.slider('Score threshold (avg_weight * number_of_doc^2):',0.0, 1.0,0.2,0.05)
    # no_pages = st.selectbox(
    # 'Number of pages:',
    # ('10', '20', '30','40','50','60','70','80','90','100','110','120','140','150'))
    no_pages = 100
    user_input = st.text_input('Keyword', 'unlock iphone')

    if  st.button("Search",no_pages) and len(user_input) > 3 :
        entity_df,words_pos, full_df =  search_keywords(user_input,no_pages)
        new_df = entity_df.groupby('entity').agg({ 'url': pd.Series.nunique,'relevanceScore': pd.Series.mean}).reset_index()
        ## sort by relevance score
        new_df = new_df.sort_values(by=['url'], ascending=False)
        st.dataframe(new_df.head(20))
        st.dataframe(full_df.head(20))

        # keywords_summary_df = ks.get_keywords_summary(entity_df, threshold, words_pos)
        # st.title('Kewords extraction using TF-IDF')
        new_df = new_df.to_csv().encode('utf-8')  
        # st.write(keywords_summary_df.sort_values('computed_score',ascending=False).head())        
        st.download_button(
        label="Download full data as CSV",
        data=new_df,
        file_name='entity_df.csv',
        mime='text/csv',
        )




if __name__ == '__main__':
    main()
