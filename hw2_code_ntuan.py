#!/usr/bin/python3
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import cmudict #counting syllables
from os import listdir, system, popen
from os.path import isfile, join, basename
from itertools import chain
import numpy, itertools
from math import log, log2, sqrt
from collections import Counter, defaultdict

# ntuan_home = "/Users/admin/Documents/cis530/hw2/data/train/"
ntuan_dict = cmudict.dict()

ntuan_home = "/home1/n/ntuan/cis530/hw2/data/train/"
ntuan_home_2 = "/home1/n/ntuan/cis530/hw2/"

def get_all_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and f.lower().endswith('.txt')]

def standardize(rawexcerpt):
    return [word for sent in sent_tokenize(rawexcerpt) for word in word_tokenize(sent)]

def load_file_excerpts(filepath):
    #read in a txt file, split by newline, standardize each line and save to a list
    #return: list of word tokens
    ret = []
    curr_file = open(filepath)
    for line in curr_file.read().split('\n'):
        ret += standardize(line)
    curr_file.close()
    return ret

def load_file_excerpts_bigram(filepath):
    #read in a txt file, split by newline, standardize each line, add beginning
    #and ending mark and save to a list
    #return: list of word tokens
    ret = []
    curr_file = open(filepath)
    for line in curr_file.read().split('\n'):
        ret += ["<s>"] + standardize(line) + ["</s>"]
    curr_file.close()
    return ret

def get_type_token_ratio(counts_file):
    num_type = 0
    num_tok = 0
    curr_file = open(counts_file)
    counts = [int(line.split(',')[-1]) for line in curr_file.read().split('\n')[:-1]]
    curr_file.close()
    return float(len(counts))/sum(counts)

def get_good_turing(frequency_model):
    count_count = defaultdict(int)
    curr_file = open(frequency_model)
    counts = [int(line.split(',')[-1]) for line in curr_file.read().split('\n')[:-1]]
    count_count = Counter(counts)
    #get # of words appearing 0 to 5 times
    #base case 0 needs N_0 = 1, case 5 needs N_6
    curr_file.close()
    count = [count_count[i] for i in range(1,7)]
    count.insert(0, 1)
    smoothed_count = defaultdict(float)
    for i in range(6):
        smoothed_count[i] = (i+1)*float(count[i+1])/count[i]
    return smoothed_count

def get_entropy(unigram_counts_file):
    ugram_i = UnigramModel(unigram_counts_file)
    return -sum([ugram_i.unigram[word_i]*ugram_i.logprob(word_i) for word_i in ugram_i.unigram])

def flatten(listoflists):
    return list(itertools.chain.from_iterable(listoflists))

# def ngram(corpus, N = 2):
#     #this function count N-grams packed into a list
#     return [" ".join(corpus[i:i+k]) for k in range(1, N+1) for i in range(len(corpus)-k+1)]

def ngram(corpus, N = 2):
    #this function count N-grams packed into a list
    return [" ".join(corpus[i:i+N]) for i in range(len(corpus)-N+1)]

#easier to understand loop version that counts everything from 1-gram to n-gram
#     for k in range(1, N+1):
#         for i in range(len(corpus) - k + 1):
#             yield " ".join(corpus[i:i+k])

class UnigramModel:
    def __init__(self, freqmodel):
        freq_model = defaultdict(int)
        self.N = 0
        self.unigram = defaultdict(float)
        curr_file = open(freqmodel)
        for line in curr_file.read().split('\n')[:-1]:
            word_i, count_i = line.rsplit(',', 1)[0], int(line.rsplit(',', 1)[1])
            freq_model[word_i] = count_i
            self.N += count_i
        curr_file.close()
        self.unigram.update({word: float(freq_model[word])/self.N for word in freq_model})

    def logprob(self, target_word):
        if self.unigram[target_word]:
            return log2(self.unigram[target_word])
        return -float("inf")

class BigramModel:
    def __init__(self, trainfiles):
        self.bigram_count = Counter()
        self.unigram_count = Counter()
        for name_i in trainfiles:
            file_i = load_file_excerpts_bigram(name_i)
            unigram_i = Counter(file_i)
            unk_i = set([word_i for word_i in unigram_i if unigram_i[word_i] == 1])
            for word in unk_i:
                unigram_i.pop(word)
            unigram_i["<UNK>"] = len(unk_i)
            file_i = ["<UNK>" if word in unk_i else word for word in file_i] #replace all N_1 word with UNK
            self.bigram_count += Counter(ngram(file_i))
            self.unigram_count += unigram_i
        self.V = len(self.unigram_count)

    def logprob(self, prior_context, target_word):
        if prior_context not in self.unigram_count:
            prior_context = "<UNK>"
        if target_word not in self.unigram_count:
            target_word = "<UNK>"
        count = self.bigram[prior_context + " " + target_word]
        unigram_count = self.unigram[prior_context]
        bigram_prob = (float(count) + 0.25)/(unigram_count + self.V*0.25)
        return log2(bigram_prob)



def load_directory_freqmodel(dirpath):
    #read all files in the dirpath
    #for each file, calculate the freqmodel and output it to a file in word,count format
    #for each file, calculate the GT count and output them by file name.
    #return: None
    vocablist = {}
    output = defaultdict(list)
    type_tok_ratio = defaultdict(float)
    entropy = []
    for file_i in get_all_files(dirpath):
        count_count = defaultdict(int)
        name_i = file_i[:-4] + "_freqmodel.txt"
        sample_i = load_file_excerpts(dirpath + file_i)
        freqmodel_i = Counter(sample_i)
        if not isfile(join(ntuan_home_2, name_i)):
            #write out the freqmodel once
            with open(ntuan_home_2 + name_i, 'w') as csvfile:
                for word_i in freqmodel_i:
                    count_i = freqmodel_i[word_i]
                    count_count[count_i] += 1
                    csvfile.writelines(word_i + ',' + str(freqmodel_i[word_i]) + '\n')

        vocab_size = len(freqmodel_i)
        entropy.append((name_i, get_entropy(ntuan_home_2+name_i)))
        type_tok = float(vocab_size)/len(sample_i)
        type_tok_ratio[file_i] = type_tok

        output[file_i].append(file_i)
        output[file_i].append(vocab_size)

        vocablist[file_i] = set(freqmodel_i.keys())

        frac_freq = float(sum([int(freqmodel_i[word_i] > 5) for word_i in freqmodel_i]))/vocab_size
        output[file_i].append(frac_freq)

        frac_rare = float(sum([int(freqmodel_i[word_i] == 1) for word_i in freqmodel_i]))/vocab_size
        output[file_i].append(frac_rare)

        word_len_dist = numpy.array([len(word_i) for word_i in sample_i])

        median_word = numpy.median(word_len_dist)
        output[file_i].append(median_word)

        average_word = numpy.mean(word_len_dist)
        output[file_i].append(average_word)

        smoothed_GT = get_good_turing(ntuan_home_2+name_i)
        if not isfile(ntuan_home_2 + "hw2_1_2_" + file_i[:-4] + ".txt"):
            with open(ntuan_home_2 + "hw2_1_2_" + file_i[:-4] + ".txt", 'w') as csvfile:
                for i in range(6):
                    csvfile.writelines(repr(count_count[i]) + '\t' + repr(smoothed_GT[i]) + '\n')

    #compare nyt vocab against other corpora
    nyt = vocablist['nytimes.txt']
    for file_i in vocablist:
        curr = vocablist[file_i]
        frac = 1 - float(len(nyt & curr))/output[file_i][1]
        output[file_i].append(frac)

    if not isfile(ntuan_home_2 + 'hw2_2_1.txt'):
        with open(ntuan_home_2 + 'hw2_2_1.txt', 'w') as fp:
            for file_i in output:
                fp.writelines(','.join(map(str, output[file_i])) + '\n')

    type_tok_ratio = list(type_tok_ratio.items())
    type_tok_ratio = sorted(type_tok_ratio, key=lambda x: x[1])
    if not isfile(ntuan_home_2 + 'hw2_2_2.txt'):
        with open(ntuan_home_2 + 'hw2_2_2.txt', 'w') as fp:
            fp.writelines(item[0] + '\n' for item in type_tok_ratio)

    entropy = sorted(entropy, key=lambda x: x[1])
    if not isfile(ntuan_home_2 + "hw2_2_3.txt"):
        with open(ntuan_home_2 + "hw2_2_3.txt", 'w') as csvfile:
            csvfile.writelines(item[0] + '\n' for item in entropy)

def output100(lm_model):
    curr_file = open(lm_model)
    with open(lm_model + "_100", 'w') as csvfile:
        csvfile.writelines("\n".join(curr_file.read().split('\n')[:100]))
    curr_file.close()

def srilm_preprocess(raw_text, temp_file):
    sents = sent_tokenize(raw_text) #list of sentences
    with open(temp_file, 'w') as csvfile:
        csvfile.writelines(sent + '\n' for sent in sents)

def srilm_bigram_models(input_file, output_dir):
    curr_file = open(input_file)
    raw_text = curr_file.read().split('\n')
    base = basename(input_file)
    if len(raw_text[-1]) == 0:
        raw_text.pop()
    raw_text = "".join(raw_text)
    curr_file.close()
    temp_i = output_dir + base + "_temp.txt"
    srilm_preprocess(raw_text, temp_i)
    uni = output_dir + base + ".uni.lm"
    bigram = output_dir + base + ".bi.lm"
    bigram_kn = output_dir + base + ".bi.kn.lm"
    if base == "nytimes.txt" and not isfile(uni + "_100"):
        system("cd /home1/c/cis530/srilm/; ngram-count -text " + temp_i + " -lm " + uni + " -order 1 -addsmooth 0.25")
        output100(uni)

    if base == "obesity.txt" and not isfile(bigram + "_100"):
        system("cd /home1/c/cis530/srilm/; ngram-count -text " + temp_i + " -lm " + bigram + " -order 2 -addsmooth 0.25")
        output100(bigram)

    if base == "cancer.txt" and not isfile(bigram_kn + "_100"):
        system("cd /home1/c/cis530/srilm/; ngram-count -text " + temp_i + " -lm " + bigram_kn + " -order 2 -kndiscount")
        output100(bigram_kn)

def srilm_ppl(model_file, raw_text):
    #assumes that user is in the SRILM directory
    temp = ntuan_home_2 + "raw_text.txt"
    srilm_preprocess(raw_text, temp)
    lcmd = "/home1/c/cis530/srilm/ngram -lm "+ model_file + " -ppl " + temp
    res = popen(lcmd).read()
    #format of the return of ngram -lm -ppl:
    #0 zeroprobs, logprob= -49389.2 ppl= 2835.18 ppl1= 2969.82
    return float(res.split("ppl")[-2].split(" ")[-2])

#list of features to consider for readability score
#Flesch-Kincaid which needs #words, #sents, #syllables
#Entropy: lower entropy means a lot of words would re-appear, making it easier to read as we have seen previous context of that word
#Avg word length: docs with longer words on average tend to be harder
#Frac nyt: since NYT is the easiest corpus in the training set, we let NYT be the baseline and docs with more words out of NYT tend to be harder
#Cosine similarity: which one among the 3 training corpora is the current line most similar to
#Strategy: Cluster into 3 class nyt, cancer, obesity, and then compare within class ?
#Alr know cancer > obesity > nyt. Now within a class, how to compare different excerpts ?

#Use all the features above...things with low entropy + low FK + high FL are easier

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

# def nsyl(word):
#     if word.lower() in ntuan_dict:
#         return max([len(list(y for y in x if y[-1].isdigit())) for x in ntuan_dict[word.lower()]])
#     else:
#         return 1

def readability(raw_text):
    #raw_text is the absolute path to the test data
    scores = []
    curr_file = open(raw_text)
    predict_class = open(ntuan_home_2 + "classes.txt")
    baseline = {"cancer":200, "obesity":150, "nytimes":100}
    corpus = curr_file.read().split('\n')
    if len(corpus[-1]) == 0:
        corpus.pop()
    classes = predict_class.read().split('\n')
    #default readability score based on the class of the excerpt, 300 for cancer, 200 for obesity, 100 for nyt
    for i in range(len(corpus)):
        line = corpus[i]
        class_i = classes[i]
        sents = sent_tokenize(line)
        total_sent_length = sum([len(sent) for sent in sents])
        num_sents = len(sents)
        tokens = flatten([word_tokenize(sent) for sent in sents])
        N = len(tokens)
        counts = Counter(tokens)
        type_tok = float(len(counts))/(N)
        unigram = {word:float(counts[word])/N for word in counts}
        #count all tokens that are not punctuations in cloze abstract
        words = [word for word in tokens if not len(word) == 1 or word.isalpha()]
        extra_long = len([word for word in words if len(word) > 13])
        word_len = numpy.array([len(word) for word in words])
        avg_word_len = numpy.mean(word_len)
        median_word_len = numpy.median(word_len)
        num_words = len(words)
        num_syl = sum([nsyl(word.lower()) for word in words])
        avg_syl_per_word = float(num_syl)/num_words
        avg_word_per_sent = float(num_words)/num_sents
        FL_score = 206.835 - 1.015*avg_word_per_sent - 84.6*avg_syl_per_word
        FK_score = 0.39*avg_word_per_sent + 11.8*avg_syl_per_word - 15.59


        # model_file = ["cancer.txt.bi.kn.lm", "nytimes.txt.uni.lm", "obesity.txt.bi.lm"]
        # ppl = [srilm_ppl(ntuan_home_2 + file_i, line) for file_i in model_file]
        # #naively classify the current excerpt to be the same as the doc with the highest ppl score
        # highest_ppl = min(ppl)
        base_score = baseline[class_i]
        entropy = -sum([unigram[word_i]*log2(unigram[word_i]) for word_i in unigram])
        print(FK_score, FL_score, entropy, type_tok, avg_word_len, median_word_len, extra_long)
        final_score = FK_score - FL_score + 3*entropy + 10*avg_word_len + 10*median_word_len + 10*type_tok + extra_long
        scores.append(repr(final_score))
    submission = ntuan_home_2 + "readability.txt"
    with open(submission, 'w') as csvfile:
        csvfile.writelines("\n".join(scores))
    curr_file.close()
    predict_class.close()

if __name__ == "__main__":
    load_directory_freqmodel(ntuan_home)
    readability(ntuan_home_2 + "data/test/cloze.txt")
    for file_i in get_all_files(ntuan_home):
        srilm_bigram_models(ntuan_home + file_i, ntuan_home_2)


# will refactor load_directory_freqmodel if time permits
