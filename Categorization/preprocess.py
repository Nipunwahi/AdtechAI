import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
dir = "Categorization/"
# directory of dataset
dataset = 'data/data.json'
class Preprocess:
    def __init__(self,datafile = dir + dataset,vocab_size = 80000):
        self.df = pd.read_json(datafile,lines =True)
        self.reduceCategory()
        self.dic,self.rev_dic,self.cls = self.createDict()
        self.df["Tag_ID"] = [self.dic[a] for a in self.df.Tag]
        self.vocab_size = vocab_size
        self.oov_tok = '<OOV>'


    def addstr(self,x,y):
        return x+ " . " + y

# it reduces and merges number of categories to 1 so that dataset is more balanced
    def reduceCategory(self):
        self.df['category_merged']=self.df['category'].replace({"HEALTHY LIVING": "WELLNESS",
        "QUEER VOICES": "GROUPS VOICES",
        "BUSINESS": "BUSINESS & FINANCES",
        "PARENTS": "PARENTING",
        "BLACK VOICES": "GROUPS VOICES",
        "THE WORLDPOST": "WORLD NEWS",
        "STYLE": "STYLE & BEAUTY",
        "GREEN": "ENVIRONMENT",
        "TASTE": "FOOD & DRINK",
        "WORLDPOST": "WORLD NEWS",
        "SCIENCE": "SCIENCE & TECH",
        "TECH": "SCIENCE & TECH",
        "MONEY": "BUSINESS & FINANCES",
        "ARTS": "ARTS & CULTURE",
        "COLLEGE": "EDUCATION",
        "LATINO VOICES": "GROUPS VOICES",
        "CULTURE & ARTS": "ARTS & CULTURE",
        "FIFTY": "MISCELLANEOUS",
        "GOOD NEWS": "MISCELLANEOUS"})
        self.df['joint'] = self.addstr(self.df['headline'],self.df['short_description'])
        self.df = self.df[['category_merged','joint']]
        self.df = self.df.rename(columns={"category_merged":"Tag","joint" : "headline"})

# create a dictionary of categories and their corresponding numbers
    def createDict(self):
        dic_tag = {}
        dic_tag_to_name = {}
        place = 0
        for tag in self.df.Tag:
            if tag not in dic_tag:
                dic_tag[tag] = place
                dic_tag_to_name[place] = tag
                place += 1

        rev_dic_tag = {v: k for k,v in dic_tag.items()}
        cls1= [i for i in range(len(dic_tag))]
        self.df["Tag_ID"] = [dic_tag[a] for a in self.df.Tag]
        return dic_tag,rev_dic_tag,cls1
    

# tokenizes the dataset based on training and maxlength number long sentences.
    def tokenize(self,train_x,val_x,max_length):
        self.tokenizer = Tokenizer(num_words = self.vocab_size,oov_token= self.oov_tok)
        self.tokenizer.fit_on_texts(train_x)
        word_index = self.tokenizer.word_index
        print(len(word_index))
        training_sequences = self.tokenizer.texts_to_sequences(train_x)
        testing_sequences = self.tokenizer.texts_to_sequences(val_x)
        padded = pad_sequences(training_sequences,maxlen = max_length,padding='post')
        test_padded = pad_sequences(testing_sequences,maxlen = max_length,padding='post')
        return padded,test_padded

    def getdf(self):
        print(self.df)
        return self.df
    
    def getTokenizer(self):
        return self.tokenizer

    

