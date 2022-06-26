import nltk
from nltk.corpus import words

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tensorflow as tf
import tflearn
import random
import json
import pickle

from time import sleep

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for patterns in intent["patterns"]:
            wrds = nltk.word_tokenize(patterns)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickles", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len((output[0])), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn3")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(words.lower()) for words in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def process(message):
    inp = message


    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.8:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        sleep(3)
        Bot = random.choice(responses)
        return (Bot)
    else:
        return ("i don't understand! \npls give me more information")



from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(process(userText))

if __name__ == "__main__":
    app.run(debug=False, host='0,0,0,0')
