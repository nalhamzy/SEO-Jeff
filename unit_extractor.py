from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import re 
import nltk


nltk.download('punkt')
nltk.download('stopwords')
from pattern.en import lexeme
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")

from nltk.tokenize import sent_tokenize
units_df = pd.read_csv('unit.csv')

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
                                curr_set = numeric_pairs[key]
                                curr_set.add(doc_num)
                        if prev_token.tag_ == "NN" or prev_token.tag_ == "NNS" or prev_token.tag_ == "SYM":
                            if prev_token.text != "-":
                                match = difflib.get_close_matches(prev_token.text.lower(), units_df['unit'].values, cutoff=0.95)
                                if len(match) > 0:
                                    numeric_df.loc[len(numeric_df)] = {'Value':token.text, 'Unit':prev_token.text,'Key':key ,'Doc':doc_num, 'Sentence':sent}
    
    unique_numeric_df = numeric_df[numeric_df.duplicated(['Key','Doc']) == False]
    unique_numeric_df = unique_numeric_df[unique_numeric_df.duplicated(['Key'])]
    
    
    for idx, row in unique_numeric_df.iterrows():
        unique_numeric_df.at[idx,'No_Documents'] = int(len(numeric_pairs[row.Key]))

    return unique_numeric_df