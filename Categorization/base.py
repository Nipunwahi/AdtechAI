from Categorization.preprocess import Preprocess
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score 
from numpy import asfarray
import pickle

dir = "Categorization/"

class Model:
# if you want to load the model, just put loaded = true
    def __init__(self,loaded = False,tosave = False):
        self.max_length = 25
        self.embed_dim = 85000
        if not loaded:
            self.split()
            self.makeModel()
            self.fit()
            if tosave:
                self.save()
        else :
            self.load()

    def makeModel(self):
        # self.clf = PassiveAggressiveClassifier()
        
        self.model = tf.keras.Sequential([
         tf.keras.layers.Embedding(self.embed_dim,100,weights = [self.embedding_matrix]),
         tf.keras.layers.LSTM(64),
         tf.keras.layers.Dense(52,activation='relu'),
         tf.keras.layers.Dense(26,activation='softmax')                  
        ])
        self.model.compile(loss = "sparse_categorical_crossentropy",optimizer = 'adam',metrics = ['accuracy'])
        print(self.model.summary())
    
    def useGlove(self):
        embeddings_dictionary = dict()
        glove_file = open('drive/My Drive/glove.6B.100d.txt')
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asfarray(records[1:],dtype = 'float32')
            embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        self.embedding_matrix = np.zeros(self.embed_dim,100)
        for word,index in self.tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector


    def split(self):
        obj = Preprocess(vocab_size = self.embed_dim)
        processedDF = obj.getdf()
        training, val = train_test_split(processedDF, test_size=0.1, random_state=42)
        self.train_x = training.headline
        self.train_y = training.Tag_ID
        self.val_x = val.headline
        self.val_y = val.Tag_ID
        self.padded,self.test_padded = obj.tokenize(self.train_x,self.val_x,self.max_length)
        self.padded = np.array(self.padded)
        self.test_padded = np.array(self.test_padded)
        self.training_label = np.array(self.train_y)
        self.test_label = np.array(self.val_y)
        self.dic,self.rev_dic,self.cls = obj.createDict()
        self.tokenizer = obj.getTokenizer()

    def fit(self):
        self.model.fit(self.padded,self.training_label,epochs = 7,validation_data=(self.test_padded,self.test_label),verbose = 1,batch_size= 512)
        # self.clf.partial_fit(self.padded,self.training_label,classes= self.cls)
        # y_pred = self.clf.predict(self.test_padded)
        # acc = accuracy_score(y_true=self.test_label,y_pred= y_pred)
        # print(acc*100)

    
    def predict(self,inp):
        seq = pad_sequences(self.tokenizer.texts_to_sequences(inp),maxlen = self.max_length,padding = 'post')
        result = self.model.predict(seq)
        print(result)
        sigma = np.std(result,axis = 1)
        mx = np.max(result,axis = 1)
        val = mx - sigma
        print(val)
        res = (result - val)>=0
        res = np.argwhere(res)
        # print(res)
        dic = {}
        for arr in res:
            dic[arr[0]] = []
        for arr in res:
            dic[arr[0]].append(self.rev_dic[arr[1]])
        # print(dic)
        res = np.argmax(self.model.predict(seq),axis = 1)
        return dic[0]
        # ans = []
        # for x in res:
        #     print(self.rev_dic[x])
        #     ans.append(self.rev_dic[x])
        # return ans

    def save(self):
        self.model.save("model.h5")
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('rev_dic.pickle', 'wb') as handle:
            pickle.dump(self.rev_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self):
        self.model = load_model(dir + 'model.h5')
        with open(dir + 'tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        with open(dir + 'rev_dic.pickle', 'rb') as handle:
            self.rev_dic = pickle.load(handle)
        # print(self.model.summary())

inp = ['hello donald trump','I like icecream',"virus attack in china","Black lives matter","pollution is decreasing"]
if __name__ == '__main__':
    model_instance = Model(loaded = False,tosave = False)
    # model_instance.load()
    model_instance.predict(inp)
    # model_instance.fit()
    # model_instance.save()
