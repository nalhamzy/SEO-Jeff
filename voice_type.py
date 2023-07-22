import spacy
from spacy.matcher import Matcher
import pandas as pd


def process_df(entity_df):
    url_df = entity_df['link'].unique()
    text_df =  entity_df['text'].unique()

    result_df = pd.DataFrame(columns=['url','text','is_passive','is_active','passive_percentage','active_percentage'])
    for i in range(len(text_df)):
        
        
        
        text = text_df[i]
        url = url_df[i]
        print(text)
        is_passive, passive_percentage, active_percentage, is_valid = isPassive(text)
        if is_valid:
            result_df.loc[len(result_df)] = {'url':url,'text':text,'is_passive':is_passive == True,'passive_percentage':passive_percentage, 'active_percentage':active_percentage}
    return result_df
def isPassive(text):

    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    text = text
    doc = nlp(text)
    sents = list(doc.sents)
    print("Number of Sentences = ",len(sents))
    # for sent in doc.sents:
    #     for token in sent:
    #         print(token.dep_,token.tag_, end = " ")
    #     print()
    passive_rule = [[{'DEP':'nsubjpass'},{'DEP':'aux','OP':'*'},{'DEP':'auxpass'},{'TAG':'VBN'}]]
    matcher.add('Passive',passive_rule)
    matches = matcher(doc)
    print(len(matches))
    print('text {}'.format(text))
    print(len(sents))
    passive_percentage = len(matches)/max(1,len(sents))
    active_percentage = 1 - passive_percentage

    return len(matches) > len(sents)/2, passive_percentage * 100, active_percentage * 100, len(sents) > 0
     
    