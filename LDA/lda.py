import nltk
import spacy
import gensim
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
from pprint import pprint
import pickle
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

path = "LDA/"
class TopicLDA:
    def __init__(self):
        self.ldamodel = gensim.models.ldamodel.LdaModel.load(path+'ldamodel.pkl', mmap='r')    #needs 3 other files in same folder too
        
        #load other files helpful for data cleaning
        self.id2word = corpora.Dictionary.load(path + 'ldamodel.pkl.id2word')
        self.datafile = open(path + 'data_words.pkl', 'rb')
        self.data_words = pickle.load(self.datafile)
        self.nlp = spacy.load("en", disable=['parser', 'ner'])
        self.bigram_mod = gensim.models.phrases.Phraser.load(path + 'bigram_mod2')

    def predict(self,text):
        words = text.split()
        
        #remove punctuation and capitals
        clean_words = gensim.utils.simple_preprocess(str(words), deacc=True)

        text_nostop = [word for word in clean_words if word not in stop_words]

        # Form Bigrams
        text_bigrams = self.bigram_mod[text_nostop]

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
       

        # Do lemmatization keeping only noun, adj, vb, adv
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']

        sent = self.nlp(" ".join(text_bigrams)) 
        text_lemmatized = []
        text_lemmatized.append([token.lemma_ for token in sent if token.pos_ in allowed_postags])
        text_lemmatized = text_lemmatized[0]

        #create bag of words
        text_bow = self.id2word.doc2bow(text_lemmatized)
        
        #find topic numbers in desc order
        topic_list = sorted(self.ldamodel.get_document_topics(text_bow), key = lambda x: (x[1]), reverse = True)

        #dictionary to name topics, made using analysis on original dataset
        topic_map = {1.0: 'American-Russian Relations', 3.0: 'Food and Restaurants', 9.0: 'Movies and Entertainment', 13.0: 'Law wrt Health and Insurance', 15.0: 'Daily City Life', 2.0: 'American President Policy and Administration', 18.0: 'Election and Politics', 26.0: 'Power and Leadership', 4.0: 'War, Crime and Police', 12.0: 'America-China Relationship', 8.0: 'Economy', 7.0: 'Fashion', 24.0: 'Business and Industry', 0.0: 'Medicine', 25.0: 'Science', 22.0: 'American Borders and Immigration', 10.0: 'Courts and Law', 11.0: 'Musuems and Art', 23.0: 'Political Speeches and Interviews', 6.0: 'Sports', 14.0: 'Jobs', 21.0: 'Travel', 19.0: 'Health, Fitness and Lifestyle', 17.0: 'People and Family Lives', 20.0: 'Internet and Technology', 27.0: 'Literature and Books', 5.0: 'Frauds, Scandals and their Investigation', 16.0: 'Community Division'}

        print('Topics: ')
        dic = {}
        dicin = {}
        for top, per in topic_list[:5]:
            dicin[topic_map[top]] = per
            print(topic_map[top], per)

        dic['topic'] = dicin

        print('Keywords: ')
        from gensim.summarization import keywords
        # text_imp = []
        # text_imp.append([token.text for token in sent if token.pos_ in allowed_postags])
        # text_imp = text_imp[0]
        # rawtext = ' '.join(text_lemmatized)
        rawtext = ' '.join(clean_words)
        print(rawtext)
        dic['keywords'] = keywords(rawtext, words = 20, split = False, scores = True, lemmatize=False)
        # dic['lemmatized'] = rawtext
        print(keywords(rawtext, words = 20, split = False, scores = True, lemmatize=False))
        return dic
    


