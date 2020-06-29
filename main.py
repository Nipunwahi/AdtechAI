from Categorization.base import Model
import flask
import flask_cors
from flask_cors import CORS
from LDA.lda import TopicLDA 
from flask import request, jsonify
import json
import re
from subjectExt.subject_extraction import finalDict,pred,finalDict_noURL

model_instance = Model(loaded = True,tosave = False)
ob = TopicLDA()

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# this will take in a url as form and return a json file for the request showing all inputs
@app.route('/api/all',methods = ['POST'])
def getAll(): 
    url = request.form['url']
    print(url)
    dic = finalDict(url)
    title = []
    tit= dic['title']
    title.append(tit)
    dic['category'] = model_instance.predict(title)
    dic['theme'] = ob.predict(tit + " " + dic['summary'])
    topiclist = dic['theme']['keywords'][0:3]
    topiclist = [t[0] for t in topiclist]
    print(topiclist)
    ff = pred(topiclist,dic['summary'])
    new_dict = {key:val for key, val in ff.items() if val >= 0.5}
    new_dict1 = {key:val for key, val in dic['named_entities'].items() if val >= 0.5}
    new_dict2 = {key:val for key, val in dic['sub_sen'].items() if val>= 0.5}
    dic['keyword_sent'] = pred(topiclist,tit + " "+ dic['summary'])
    dic2 = {}
    dicc = {}
    dicn = {}
    x = []
    x = findUncommon(new_dict,new_dict1)
    x = findUncommon(new_dict2,x)
    print(x)
    a = getkeystring(x)
    dicn['title'] = tit
    dicn['category'] = dic['category']
    ss = a
    dicn['keywords'] = ss
    dicc['content'] = dicn
    dic2['site'] = dicc
    dic2['actual'] = dic
    return jsonify(dic2)


# this uses title and summary as input and gives the same data as /api/all
@app.route('/api/alldata',methods = ['POST'])
def gettitsum():
    title = request.form['title']
    summary = request.form['summary']
    dic = finalDict_noURL(title,summary)
    title = []
    tit= dic['title']
    title.append(tit)
    dic['category'] = model_instance.predict(title)
    dic['theme'] = ob.predict(tit + " " + dic['summary'])
    topiclist = dic['theme']['keywords'][0:3]
    topiclist = [t[0] for t in topiclist]
    print(topiclist)
    ff = pred(topiclist,dic['summary'])
    for ke in ff:
        print(ke[1],type(ke))
    dic['keyword_sent'] = [ke for ke in ff and ke[1]!=-1]
    dic2 = {}
    dicc = {}
    dicn = {}
    x = []
    x = findUncommon(dic['keyword_sent'],dic['named_entities'])
    x = findUncommon(dic['sub_sen'],x)
    print(x)
    a = getkeystring(x)
    dicn['title'] = tit
    dicn['category'] = dic['category']
    ss = a
    dicn['keywords'] = ss
    dicc['content'] = dicn
    dic2['site'] = dicc
    dic2['actual'] = dic
    return jsonify(dic2)


# returns a comma separated string which will be used for openRTB
def getkeystring(keywords):
    s = ""
    a = []
    for j in keywords:
        a.append(j)
    s = ",".join(a)
    return s

# returns set of all words in
def findUncommon(arr1,arr2):
    if len(arr2) == 0:
        return arr1
    else :
        x = []
        for el in arr1:
            print(el)
            if el not in arr2:
                print(el)
                ok = True
                for ele in arr2:
                    if el in ele:
                        ok = False
                        break
                if ok:
                    x.append(el)
        x.extend(arr2)
        return x
CORS(app)
app.run()
