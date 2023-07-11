from datasets import load_dataset
import seaborn as sns
import pandas as pd
import contractions
from matplotlib import pyplot as plt
import plotly.express as px
import plotly

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

import pandas as pd
import numpy as np
import nltk
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

label_names = {'0':'admiration', 
               '1':'amusement',
               '2':'anger',
               '3':'annoyance',
               '4':'approval',
               '5':'caring', 
               '6':'confusion',
               '7':'curiosity',
               '8':'desire',
               '9':'disappoinment',
               '10':'dissaproval', 
               '11':'disgust',
               '12':'embarrassment',
               '13':'excitement',
               '14':'fear',
               '15':'gratitude', 
               '16':'grief',
               '17':'joy',
               '18':'love',
               '19':'nervousness',    
               '20':'optimism',
               '21':'pride',
               '22':'realization',
               '23':'relief',
               '24':'remorse',
               '25':'sadness',
               '26':'surprise',
               '27':'neutral'}

label_names_merged = {'0':'Anger', 
               '1':'Sadness',
               '2':'Admiration',
               '3':'Gratitude',
               '4':'Fear',
               '5':'Surprise', 
               '6':'Joy',
               '7':'Confusion',
               '8':'Curiosity',
               '9':'Desire',
               '10':'Disgust', 
               '11':'Relief',
               '12':'Remorse',
               '13':'Neutral'}


label_names_swap = {0 : 2,   #admiration
               1 : 6,  #amusement
               2 : 0,   #anger
               3 : 0,   #annyance
               4 : 2,   #approval
               5 : 3,  #caring
               6 : 7,   #confusion
               7 : 8,   #curiosity
               8 : 9,   #desire
               9 : 1,  #dissapoinment
               10 : 1, #dissaproal
               11 : 10, #disgust
               12 : 1, #embarrassment
               13 : 6, #excitement
               14 : 4, #fear
               15 : 3, #gratitude
               16 : 1, #grief
               17 : 6, #joy
               18 : 3, #love
               19 : 4, #nervousness     
               20 : 6, #optimism
               21 : 3, #pride
               22 : 5, #realization
               23 : 11, #relief
               24 : 12, #remorse
               25 : 1, #sadness
               26 : 5, #surprise
               27 : 13} #neutral

### DATA EXPLORATION ###

def count_class_distribution(df, WITH_MULTIPLE_LABELS = True):
    counter = 0
    def add_idx(idx):
        try:
            emotion_counts[str(idx)] +=1
        except:
            emotion_counts[str(idx)] = 1

    emotion_counts = {}
    
    for index, row in df.iterrows():
        if WITH_MULTIPLE_LABELS:
            for idx in row['labels']:
                add_idx(idx)
        else:
            if len(row['labels']) == 1:
                add_idx(row['labels'][0])
    return emotion_counts

### MERGING LABELS ###

def replace_label(label):
    replace = label_names_swap.get(label)
#     print(f"{label} --> {replace}")
    return replace

def merging_labels(df_):
    df = df_.copy()
    for index, row in df.iterrows():
        for index2, label in enumerate(row['labels']):
            row['labels'][index2] = replace_label(label)
    return df



### CONTROLLING DUPLICATES

# def remove_duplicates(array):
#     return list(set(array)) 


# def controling_duplicates(df):
#     for index, row in df.iterrows():
# #         print(index)
# #         print(row['text'])
# #         print(row['labels'])
#         lista = remove_duplicates(row['labels'])
# #         print(len(lista),lista)
#         if len(lista) > 1:
#             for each in lista:
#                 row_to_duplicate = df.iloc[index:(index+1)].copy()
#                 row_to_duplicate['labels'] = [[each]]
#                 df = pd.concat([df, row_to_duplicate], ignore_index=True)

#     for index, row in df.iterrows():
#         lista = remove_duplicates(row['labels'])
#         if len(lista) > 1:
#             df = df.drop(index) 

#     return df
#############################
### EXPANDING CONTRACTIONSs###
#############################

def expanding_contractions(df):
    df['text_cont_str'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    return df

def to_string(df):
    df['text_cont_str'] = [' '.join(map(str,l)) for l in df['text_cont_str']]
    return df


###### FOREIGN LANGUAGE DETECTION ###### 

def get_lang_detector(nlp, name):
    return LanguageDetector()

def detect_language(nlp,temp, THRESHOLD):
    doc = nlp(temp)
    score = doc._.language['score']
    if score > THRESHOLD:
#         print(temp)
        return "en"
    else:
        print(score)
        print(temp)
        return "unk"
    
###### TOKENISATION ###### 

def tokenisation(df, new_column, to_tokenize):
    df[new_column] = df[to_tokenize].apply(word_tokenize)
    return df
    
###### LOWERCASE FUNCTIONS ###### 

def to_lowercase(df, new_column, to_lowercase):
    df[new_column] = df[to_lowercase].apply(lambda x: [word.lower() for word in x])

    
    
###### REMOVING PUNCTUATIONS ###### 
punc = string.punctuation

def removing_punctuation(df, new_column, to_tokenize):
    df[new_column] = df[to_tokenize].apply(lambda x: [word for word in x if word not in punc])
    return df



###### STOPWORDS REMOVAL ##### 
stop_words = set(stopwords.words('english'))

def removing_stopwords(value, new_column, to_process):
    value[new_column] = value[to_process].apply(lambda x: [word for word in x if word not in stop_words])
    
    
    
###### POSITIONAL TAGGING ###### 
stop_words = set(stopwords.words('english'))

def pos_tagging(value, new_column, to_process):
    value[new_column] = value[to_process].apply(nltk.tag.pos_tag)

