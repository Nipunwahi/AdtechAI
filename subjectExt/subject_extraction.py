# -*- coding: utf-8 -*-
"""Copy of subject_final_modified.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E9Hnfs6ZT_RGVlVtVLUJo9UDoyOIzShL

Sentiment Analysis Model (s in the name of sentiment analysis object)
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle
import re

dir = "subjectExt/"

# class SAModel:
#     def __init__(self):
#         self.model=tf.keras.models.load_model(dir + 'Training/SA_Model')
#         with open(dir + 'Training/tokenizer.pickle', 'rb') as handle:
#             self.tokenizer = pickle.load(handle)
            
#     def predict(self,inp):        #can give a single string as input, returns 0 for negitive and 1 for positive
#         inp = [inp]
#         inp = self.tokenizer.texts_to_sequences(inp)
#         inp = pad_sequences(inp, maxlen=100, dtype='int32', value=0)
#         sentiment = self.model.predict(inp,batch_size=1,verbose = 0)[0]
#         return np.argmax(sentiment)
    
#     def predictByTopic(self,topic,inp):    #Returns if inp is positive or negitive to topic, inp is a string
#         inp = inp.split('. ',-1)
#         sum = 0
#         o = 0
#         z = 0
#         l = 0
#         for s in inp:
#             s = s.lower()
#             if topic in s:
#                 l=l+1
#                 inp = s
#                 inp = [inp]
#                 inp = self.tokenizer.texts_to_sequences(inp)
#                 inp = pad_sequences(inp, maxlen=100, dtype='int32', value=0)
#                 sentiment = self.model.predict(inp,batch_size=1,verbose = 0)[0]
#                 sum=sum+sentiment[1]
#                 if(sentiment[1]<0.5):
#                     z=z+1
#                 else:
#                     o=o+1
#         try: 
#           x = sum/l
#         except:
#           x = -1
#         #print(f'positive sentenses:{o}, negitive sentenses:{z}')
#         return x

#     def predictByArray(self,topics,inp):
#       ans = dict()
#       for t in topics:
#         ans[t]=self.predictByTopic(t,inp)
#       return ans
from subjectExt.sa_model import SAModel
s = SAModel(True)

"""Dependencies to be installed"""

# neuralcoref is not compatible with latest version of spacy
# spacy needs to be downgraded to the version specified below



"""Coreference Resolution"""

import spacy
nlp = spacy.load('en')
import neuralcoref
neuralcoref.add_to_pipe(nlp,greedyness = 0.527)

def coref_resol(doc):
    o = nlp(doc)
    return o._.coref_resolved

"""Some NLTK resources that are required"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords
stop = stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def word_freq_dist(document):
    words = nltk.tokenize.word_tokenize(document)
    words = [word.lower() for word in words if word not in stop]
    fdist = nltk.FreqDist(words)
    return fdist

# Returns first three of most_frequent words
def extract_subject(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(10)
                       if nltk.pos_tag([w])[0][1] in NOUNS and len(w)>2]
    return most_freq_nouns[:3]

"""Code for extracting info using newsaper"""

from newspaper import Article

# Returns coreference resolved text from a website
def webScrape(url):
  article = Article(url)
  article.download()
  article.parse()
  article.nlp()
  print(article.keywords)
  doc = ''.join([s for s in article.text.splitlines(True) if s.strip('\r\n')])
  doc = coref_resol(doc)
  doc = re.sub('[^A-Za-z .-]+', ' ' , doc)
  doc = ' '.join(doc.split())
  doc = doc.replace('\n',' ')
  return doc

#Returns keywords of a webpage
def getKeywords(url):
  article = Article(url)
  article.download()
  article.parse()
  article.nlp()
  return (article.keywords)

#returns summary (coreference resolved)
def getSummary(url):
  article = Article(url)
  article.download()
  article.parse()
  doc = ''.join([s for s in article.text.splitlines(True) if s.strip('\r\n')])
  # doc = re.sub('[^A-Za-z .-]+', ' ' , doc)
  doc = ' '.join(doc.split())
  doc = doc.replace('\n',' ')
  doc = coref_resol(doc)
  doc = doc.replace('\n',' ')
  article.text = doc
  article.nlp()
  return (article.summary.replace('\n',' '))

def getTitle(url):
  article = Article(url)
  article.download()
  article.parse()
  return article.title

# Returns a dictionary, takes URL as an input
def getSubDict(d):
  doc = (d.lower())
  subs = extract_subject(doc)
  return (s.predictByArray(subs, doc))

def getNamedEntities(d, sub_sen):
  subs = sub_sen.keys()
  document = d
  doc = nlp(document)
  store_ent = dict()
  for ent in doc.ents:
    if 'summary' not in ent.text.lower(): 
      st = ent.text.lower() 
      if(ent.text.lower()[:4] == 'the '):
        st = st[4:len(st)]
      if(st[:3] == 'mr '):
        st = st[3:len(st)]
      store_ent[st]=0
      
  ent_list = store_ent.keys()
  final_ent = dict()
  for ent in ent_list:
    for s in subs:
      if s in ent.lower():
        final_ent[ent.lower()]=sub_sen[s]
        break
  fl = final_ent.keys()
  for i in fl:
    for j in fl:
      if j in i and i!=j:
        final_ent[j]=-1
  fin_ent=dict()
  for f in fl:
    if final_ent[f]!=-1:
      fin_ent[f]=final_ent[f]
  return fin_ent


def finalDict(url):
  suggestion = dict()

  suggestion['title'] = getTitle(url)

  suggestion['summary'] = getSummary(url)
  
  suggestion['sub_sen'] = getSubDict(suggestion['summary'])

  suggestion['named_entities'] = getNamedEntities(suggestion['summary'], suggestion['sub_sen'])
  
  return suggestion

def finalDict_noURL(title,summary):
  suggestion = dict()
  suggestion['title'] = title
  suggestion['summary'] = coref_resol(summary)
  suggestion['sub_sen'] = getSubDict(suggestion['summary'])
  suggestion['named_entities'] = getNamedEntities(suggestion['summary'], suggestion['sub_sen'])
  return suggestion

# url = 'https://economictimes.indiatimes.com/news/politics-and-nation/narendra-modi-finds-neighbors-silent-as-india-china-tensions-simmer/articleshow/76458627.cms'
# url = 'https://www.ndtv.com/world-news/complete-decoupling-from-china-remains-an-option-says-donald-trump-2248643'
# # url = 'https://www.businessghana.com/site/news/politics/215687/Trump-s-bid-to-end-Obama-era-immigration-policy-ruled-unlawful'
# x = (finalDict(url))

def pred(topic,inp):
    return s.predictByArray(topic,inp)

