import nltk, re # NLTK library of language resources
nltk.download('omw-1.4')

# Part of speech tagging and tokenisation
from nltk import pos_tag, word_tokenize 

# to perform lemmatisation
from nltk.stem import wordnet, WordNetLemmatizer

# stopwords
from nltk.corpus import stopwords

import json

# to perform bow and tfidf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# For cosine similarity
from sklearn.metrics import pairwise_distances
# Data processing and visualisation
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
pd.set_option('display.precision', 3)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_excel("dialog_talk_agent.xlsx")
df.ffill(axis = 0, inplace = True)
df1 = df.head(10)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
def normalise_text(text):
    # convert to lowercase
    text = str(text).lower()
        
    # remove special characters
    text = re.sub(r'[^a-z0-9]', " ",text)
        
    # tokenise
    tokens = word_tokenize(text)
    
    # Initialise lemmatiser
    lemmatiser = wordnet.WordNetLemmatizer()
    
    # Part of speech (POS) tagging, tagset set to default
    tagged_tokens =  pos_tag(tokens, tagset = None)

    # Empty list
    token_lemmas = []
    for (token, pos_token) in tagged_tokens:
        if pos_token.startswith("V"): # verb
            pos_val = "v"
        elif pos_token.startswith("J"): # adjective
            pos_val = "a"
        elif pos_token.startswith("R"): # adverb
            pos_val = "r"
        else:
            pos_val = 'n' # noun
        
        # lemmatise and append to list of lemmatised tokens
        token_lemmas.append(lemmatiser.lemmatize(token, pos_val))
    
    return " ".join(token_lemmas)
# apply the normalise_text function to each entry in the context column
df["lemmatised_text"] = df["Context"].apply(normalise_text)
def remove_stopwords(text):
    
    # stopwords
    stop = stopwords.words("english")
    
    #if token not in stop
    text = [word for word in text.split() if word not in stop]
    return " ".join(text)
# count vectoriser 
cv = CountVectorizer()
X = cv.fit_transform(df["lemmatised_text"]).toarray()
features = cv.get_feature_names_out()
df_bow = pd.DataFrame(X, columns = features)
df_bow.head()
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# Download necessary NLTK resources
nltk.download('stopwords')

def remove_stopwords(text):
    # stopwords
    stop = stopwords.words("english")
    
    # Remove stopwords from text
    text = [word for word in text.split() if word not in stop]
    return " ".join(text)

def normalise_text(text):
    # convert to lowercase
    text = str(text).lower()
        
    # remove special characters
    text = re.sub(r'[^a-z0-9]', " ",text)
        
    # tokenise
    tokens = word_tokenize(text)
    
    # Initialise lemmatiser
    lemmatiser = wordnet.WordNetLemmatizer()
    
    # Part of speech (POS) tagging, tagset set to default
    tagged_tokens =  pos_tag(tokens, tagset = None)

    # Empty list
    token_lemmas = []
    for (token, pos_token) in tagged_tokens:
        if pos_token.startswith("V"): # verb
            pos_val = "v"
        elif pos_token.startswith("J"): # adjective
            pos_val = "a"
        elif pos_token.startswith("R"): # adverb
            pos_val = "r"
        else:
            pos_val = 'n' # noun
        
        # lemmatise and append to list of lemmatised tokens
        token_lemmas.append(lemmatiser.lemmatize(token, pos_val))
    
    return " ".join(token_lemmas)
# Initialise sklearn tfidf
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(df["lemmatised_text"]).toarray()
df_tfidf = pd.DataFrame(x_tfidf, columns = tfidf.get_feature_names_out())
df_tfidf.head()
s = "Do you like mangoes?"
#print("Starting chat_tfidf function")
def chat_tfidf(text):
    # Lemmatised utterance
    text = normalise_text(text)
    #print("Normalized Text:", text)
    
    # Transform input text to TF-IDF
    text_tfidf = tfidf.transform([text]).toarray()
    #print("TF-IDF Vector of Input Text:", text_tfidf)
    
    # Compute cosine similarity
    cos = 1 - pairwise_distances(df_tfidf, text_tfidf, metric = "cosine")
    #print("Cosine Similarities:", cos)
    
    # Get the index of the most similar text
    index_value = cos.argmax()
    #print("Most Similar Text Index:", index_value)
    
    return df["Text Response"].loc[index_value]
