from flask import Flask, request, Response, send_file, render_template, flash, redirect, url_for, jsonify
import io
from tempfile import NamedTemporaryFile


import nltk
import numpy as np
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import wordnet
nltk.download('wordnet')


stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
     return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/findtopic', methods=['POST'])
def findtopic():
        text = request.form['text']
        data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
        data = data.iloc[0:5000,0:].values
        data=pd.DataFrame(data,columns=["publish_date","headline_text"])
        data_text = data['headline_text']
        data_text['index'] = data_text.index
        documents = data_text
        data_text = data_text.drop(data_text.index[len(data_text)-1])
        documents = documents.drop(documents.index[len(documents)-1])

        doc_sample = documents[documents.index[40]]
        print('original document: ')
        words = []
        for word in doc_sample.split(' '):
                words.append(word)
        

        documents=pd.DataFrame(documents)


        processed_docs = documents['headline_text'].map(preprocess)
        print(processed_docs[:10])

        dictionary = gensim.corpora.Dictionary(processed_docs)
        count = 0
        for k, v in dictionary.iteritems():
                print(k, v)
                count += 1
                if count > 10:
                        break

        a = preprocess(text)
        print (a)
        other_corpus = [dictionary.doc2bow(a)]
        unseen_doc = other_corpus[0]
        print(unseen_doc)

        #dictionary = dictionary.filter_extremes(no_below=15, no_above=0.5)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        bow_corpus[40] 
        bow_doc_40 = bow_corpus[40]
        for i in range(len(bow_doc_40)):
                print("Word {} (\"{}\") appears {} time.".format(bow_doc_40[i][0], 
                                                        dictionary[bow_doc_40[i][0]], 
                                                                bow_doc_40[i][1]))

        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=1, id2word=dictionary)

        vector = lda_model[unseen_doc]
        print("new text")
        output = ""
        for index, score in sorted(vector,key=lambda tup: -1*tup[1]):
            output = output + ("Score:{}, \n Topic: {}".format(score, lda_model.print_topic(index,10)))
        return output




if __name__ == '__main__':    
    #app.run(debug=True)
    app.run(host='0.0.0.0')
# app.run(debug=True)
