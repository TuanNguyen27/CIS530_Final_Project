#!/usr/bin/python3
import string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from os import listdir, system, popen
from os.path import isfile, join, basename
from itertools import chain
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVR
from nltk.util import ngrams
from math import log, log2
import itertools
from collections import Counter, defaultdict
from nltk.tree import Tree
# import pdb
import re
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import os

# ntuan_home = "/Users/admin/Documents/cis530/final/"
ntuan_home = ""
# data_path = "/Users/admin/Documents/cis530/final/"
data_path = ""
train_path = data_path + "data/project_train.txt"
test_path = data_path + "data/project_test.txt"
label_path = data_path + "data/project_train_scores.txt"
optional_train_path = data_path + "data/optional_training"
optional_labels = data_path + "data/optional_training/optional_project_train_scores.txt"

#Credit: http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
ntuan_numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = ntuan_numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_all_files(directory):
    return sorted([join(directory, f) for f in listdir(directory)], key = numericalSort)

def standardize(rawexcerpt):
    return [w for w in word_tokenize(rawexcerpt.lower())]

def load_file_excerpts(filepath):
    excerpts = []
    file_i = open(filepath)
    for line in file_i.read().strip().split('\n'):
        excerpts.append(standardize(line.strip()))
    file_i.close()
    return excerpts

def flatten(listoflists):
    return list(chain.from_iterable(listoflists))

def load_optional_train(optional_train_files):
    excerpts = []
    for path_i in optional_train_files:
        file_i = open(path_i)

def nsyl(word):
    vowels = "aeiouy"
    numVowels = 0
    lastWasVowel = False
    for wc in word:
        foundVowel = False
        for v in vowels:
            if v == wc:
                if not lastWasVowel: numVowels+=1   #don't count diphthongs
                foundVowel = lastWasVowel = True
                break
        if not foundVowel:  #If full cycle and no vowel found, set lastWasVowel to false
            lastWasVowel = False
    if len(word) > 2 and word[-2:] == "es": #Remove es - it's "usually" silent (?)
        numVowels-=1
    elif len(word) > 1 and word[-1:] == "e":    #remove silent e
        numVowels-=1
    return numVowels

def build_feature(train_data_path, stopwords, nytimes):
    curr_file = open(train_data_path)
    train_data = curr_file.read().strip().split('\n')
    training_num = len(train_data)
    feature_vec = []
    for i in range(training_num):
        feature_vec_i = []
        excerpt = train_data[i]
        sents = sent_tokenize(excerpt)
        total_sent_length = sum([len(sent) for sent in sents])
        num_sents = len(sents)

        #count # words in each excerpt
        words = flatten([word_tokenize(sent) for sent in sents])
        words = [word for word in words if word not in stopwords]
        #words = excerpt

        #word count frequency
        counts = Counter(words)
        num_words = len(words)

        #unigram distrbution
        unigram = {word:float(counts[word])/num_words for word in counts}
        entropy = -sum([unigram[word_i]*log2(unigram[word_i]) for word_i in unigram])

        #words with length > 13 chars
        extra_long = len([word for word in words if len(word) > 13])

        #type-token ratio
        type_tok = len(set(words))/len(words)

        #percent of words not in nyt vocab
        nyt_frac = len([word for word in words if word not in nytimes])/len(words)

        #word length distribution
        word_len = np.array([len(word) for word in words])

        #avg length of a word in excerpt
        avg_word_len = np.mean(word_len)

        #median length of a word in excerpt
        median_word_len = np.median(word_len)

        #following steps do not work because excerpt is alr tokenized before being sent_tokenize
        num_syl = sum([nsyl(word.lower()) for word in words])
        avg_syl_per_word = float(num_syl)/num_words
        avg_word_per_sent = float(num_words)/num_sents
        FL_score = 206.835 - 1.015*avg_word_per_sent - 84.6*avg_syl_per_word

        feature_vec.append([entropy, extra_long, type_tok, nyt_frac, avg_word_len, median_word_len, FL_score])

    return np.asarray(feature_vec)


train_data = load_file_excerpts(train_path)
test_data = load_file_excerpts(test_path)
# optional_train_files = get_all_files(optional_train_path)[:-1]

curr_file = open(ntuan_home + "stopwords.txt")
stopwords = set(curr_file.read().strip().split('\n'))
curr_file.close()

#build a vocab of all words in nytimes.txt, excluding stopwords
nytimes = flatten(load_file_excerpts(ntuan_home + "nytimes.txt"))
nytimes = set([word_i for word_i in nytimes if word_i not in stopwords])
nytimes = set(nytimes)
curr_file.close()

X_train = build_feature(train_path, stopwords, nytimes)
X_test  = build_feature(test_path, stopwords, nytimes)
labels_file = open(label_path)
Y_train = np.asarray([int(line_i) for line_i in labels_file.read().strip().split('\n')])
labels_file.close()

clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
print(pred)

# labels_file = open(optional_labels)
# optional_labels = {line_i.split()[0]: (1-float(line_i.split()[1])) for line_i in labels_file.read().strip().split('\n')}
# labels_file.close()
# print(optional_labels)


# if __name__ == "__main__":
