from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import csv
import timeit
import random

def talk(user_sentence):
    csv_file_path = "./DataSet/randychat.csv"
    
    tfidf_vectorizer_pikle_path = "./DataSet/tfidf_vectorizer.pickle"
    tfidf_matrix_train_pikle_path = "./DataSet/tfidf_matrix_train.pickle"
    
    i = 0
    sentences = []
    test_set = (user_sentence,"")
    
    sentences.append(" No you")
    sentences.append(" No you")
    
    try:
        f = open(tfidf_vectorizer_pikle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()
        
        f = open(tfidf_matrix_train_pikle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        f.close()
        
    except:
        start = timeit.default_timer()
        
        with open(csv_file_path, "r") as sentence_file:
            reader = csv.reader(sentence_file, delimiter=",")
            for row in reader:
                sentences.append(row[0])
                i =+ 1
                
        tfidf_vectorizer = TfidfVectorizer()
        
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)
        
        stop = timeit.default_timer()
        #print(f"training time took was: {stop - start}")
        
        f = open(tfidf_vectorizer_pikle_path, 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()
        
        f = open(tfidf_matrix_train_pikle_path, 'wb')
        pickle.dump(tfidf_matrix_train, f)
        f.close()
        
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)
    
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    cosine = np.delete(cosine, 0)
    
    maxi = cosine.max()
    response_index = 0
    
    if (maxi > 0.7):
        new_max = maxi - 0.01
        listi = np.where(cosine > new_max)
        response_index = random.choice(listi[0])
        
    else:
        response_index = np.where(cosine == maxi)[0][0] + 2
    j = 0
    
    with open(csv_file_path,"r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=",")
        for row in reader:
            j += 1
            if j == response_index:
                return row[1], response_index,
                break


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(talk(userText)[0])


if __name__ == "__main__":
    app.run()