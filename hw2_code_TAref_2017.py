from nltk import FreqDist, ConditionalFreqDist, sent_tokenize, word_tokenize
import os, subprocess
import math, operator
import numpy as np

TRAIN_PATH = "/home1/c/cis530/hw2/data/train/"
TEST_DATA = "/home1/c/cis530/hw2/data/test/excerpts.txt"
NGRAM_COUNT = '/home1/c/cis530/srilm/ngram-count'
NGRAM = '/home1/c/cis530/srilm/ngram'

TEMPDIR = '/home1/c/cis530/temp'

NYT_CORPUS = set()
#########################
# Section 1.1
#########################

class UnigramModel:

    def __init__(self, freqmodel):
        self.unigram_model = {}
        self.total_words = 0
        with open(freqmodel, "r") as f:
            for line in f:
                l = line.strip().split(",")
                v = l[:-1]
                c = int(l[-1])
                self.unigram_model[",".join(v)] = c
                self.total_words += c

    def logprob(self, target_word):
        lprob_word = math.log(float(self.unigram_model[target_word])/self.total_words,2)
        return lprob_word

    def prob(self, target_word):
        return float(self.unigram_model[target_word])/self.total_words

#########################
# Section 1.2
#########################
def get_good_turing(freqmodel):
    nclass = {}
    N_0 = 1
    with open(freqmodel, "r") as f:
        for line in f:
            l = line.strip().split(",")
            c = int(l[-1])
            if c in nclass:
                nclass[c] = nclass[c] + 1
            else:
                nclass[c] = 1
    nclass[0] = 1
    N_1 = nclass[1]
    N_2 = nclass[2]
    N_3 = nclass[3]
    N_4 = nclass[4]
    N_5 = nclass[5]
    N_6 = nclass[6]
    r = []
    r.append(float(N_1)/N_0)
    r.append((1+1.0)*(float(N_2)/N_1))
    r.append((2+1.0) * (float(N_3) / N_2))
    r.append((3+1.0)*(float(N_4)/N_3))
    r.append((4+1.0) * (float(N_5) / N_4))
    r.append((5+1.0) * (float(N_6) / N_5))
    final_nclass = {e:val for e,val in enumerate(r)}
    with open(freqmodel + "_1_2.txt", "w") as f:
        f.write(str(nclass[0])+ "," + str(r[0]) + "\n")
        f.write(str(nclass[1]) + "," + str(r[1])+ "\n")
        f.write(str(nclass[2] )+ "," + str(r[2])+ "\n")
        f.write(str(nclass[3]) + "," + str(r[3])+ "\n")
        f.write(str(nclass[4]) + "," + str(r[4])+ "\n")
        f.write(str(nclass[5] )+ "," + str(r[5]))
        f.close()
    return final_nclass

#########################
# Section 2.1
#########################
def generate_nytimes_set(nyt):
    with open(nyt, 'rU') as ny:
        for line in ny:
            val = line.split(",")
            NYT_CORPUS.add(','.join(val[:-1]))
def get_vocab_stats(countsfile):
    filename = countsfile.split("_")[0]
    total = 0
    total_unique = 0
    rare = 0
    freq = 0
    med_len = []
    not_in_times = 0
    average_word = 0
    with open(countsfile, 'rU') as fin:
        #w,c
        generate_nytimes_set("nytimes_freq.txt")
        for line in fin:
            total_unique += 1
            val= line.split(",")
            cnt = val[-1]
            wrd = ','.join(val[:-1])
            if wrd not in NYT_CORPUS:
                not_in_times += 1
            cnt = int(cnt)
            if cnt > 5:
                freq += 1
            if cnt == 1:
                rare += 1
            total += cnt
            for i in range(cnt):
                average_word += len(wrd)
            med_len.extend([len(wrd) for i in range(cnt)])
        med_len.sort()
        mid =  len(med_len) / 2
        if len(med_len)%2 == 0:
            mid_1 = mid-1
            med = (float(med_len[mid]) + float(med_len[mid_1]))/2
        else:
            med = float(med_len[mid])

    return filename, total_unique,float(freq)/total_unique,float(rare)/total_unique,med,float(average_word)/total,(float(not_in_times)/total_unique)
# vocab size: the number of unique tokens in the vocabulary for each excerpt type;
# frac freq: the fraction of words in the vocabulary appearing more than 5 times;
# frac rare: the fraction of words in the vocabulary appearing only once;
# median word: The length (in characters) of the median word in the text;
# average word: The length (in characters) of the average word in the text;
# frac nyt: The fraction of the vocabulary that is not in the New York Times domain.



#########################
# Section 2.2
#########################
def get_type_token_ratio(freq_file):
    types = 0
    tokens = 0

    with open(freq_file, 'rU') as fin:
        for line in fin:
            val = line.split(',')
            cnt = val[-1]
            types += 1
            tokens += int(cnt)

    return float(types) / tokens
#########################
# Section 2.3
#########################
def get_entropy(freqfile):
    probs = []
    uni_model = UnigramModel(freqfile)
    for word in uni_model.unigram_model:
        probs.append(float(uni_model.prob(word)))
    print sum(probs)
    entropy = 0.0
    for p in probs:
        p_w =  p
        p_2_w = math.log(p_w, 2)
        entropy -= p_w * p_2_w
    return entropy

#########################
# Section 3
#########################

def read_sents(filename):
    '''
    Read in file containing sentences and some newlines. Sentence tokenize and
    return a list of sentences.
    :param filename: string filename
    :return: list of strings
    '''
    data = open(filename).read().decode('utf8').strip().replace('\n', ' ')
    sents = sent_tokenize(data)
    return sents

def flatten(listoflists):
    return [item for sublist in listoflists for item in sublist]


def make_bigram_tuples(word_list):
    return [tuple(word_list[x:x+2]) for x in xrange(len(word_list)-1)]


class BigramModel:
    def __init__(self, trainfiles):
        # Read files and tokenize
        tokens = []
        for i in trainfiles:
            sentences = read_sents(i)
            tokens.extend(flatten([['<s>'] + word_tokenize(s) + ['</s>'] for s in sentences]))
        # Count tokens and replace those occuring once with '<unk>'
        tok_counts = FreqDist(tokens)
        tokens = [t if tok_counts[t] > 1 else '<unk>' for t in tokens]
        # Create bigram list
        bigrams = make_bigram_tuples(tokens)

        # Record counts
        self.cnt_toks = FreqDist(tokens)
        self.vocab = set(self.cnt_toks.keys())
        self.V = len(self.vocab)
        self.k = 0.25
        self.cnt_bigrams = FreqDist(bigrams)

    def logprob(self, prior_context, target_word):
        newcontext = '<unk>' if prior_context not in self.vocab else prior_context
        newtarget = '<unk>' if target_word not in self.vocab else target_word
        lp = math.log((self.cnt_bigrams[(newcontext, newtarget)] + self.k) /
                      (self.cnt_toks[newcontext] + self.k * self.V), 2)
        return lp

#########################
# Section 4
#########################

def srilm_preprocess(raw_file, temp_file):
    outf = open(temp_file, 'w')
    sents = read_sents(raw_file)
    for sent in sents:
        outf.write(sent.strip().encode("utf8")+"\n")
    outf.close()

def srilm_preprocess_text(raw_text, temp_file):
    outf = open(temp_file, 'w')
    txt = raw_text.strip().replace('\n', ' ').decode('utf8')
    sents = sent_tokenize(txt)
    for sent in sents:
        print >> outf, sent.strip().encode('utf8')
    outf.close()


def srilm_bigram_models(input_file, output_dir):
    basename = os.path.basename(input_file)

    # Temporarily output to SRILM format
    if not os.path.exists(TEMPDIR):
        os.mkdir(TEMPDIR)
    tempfname = os.path.join(TEMPDIR, 'temp')
    srilm_preprocess(input_file, tempfname)

    # Create SRILM language models
    uni_addk_filename = os.path.join(output_dir, basename+'.uni.lm')
    uni_addk_countsname = os.path.join(output_dir, basename+'.uni.counts')
    os.system('%s -write %s -order 1 -addsmooth 0.25 -text %s -lm %s'
              %(NGRAM_COUNT, uni_addk_countsname, tempfname, uni_addk_filename))

    bi_addk_filename = os.path.join(output_dir, basename+'.bi.lm')
    os.system('%s -order 2 -addsmooth 0.25 -text %s -lm %s'
              %(NGRAM_COUNT, tempfname, bi_addk_filename))

    bi_kn_filname = os.path.join(output_dir, basename+'.bi.kn.lm')
    os.system('%s -order 2 -kndiscount -text %s -lm %s'
              %(NGRAM_COUNT, tempfname, bi_kn_filname))

    # Cleanup temp file
    os.remove(tempfname)


def srilm_ppl(model_file, raw_text):

    # Temporarily output raw text to SRILM format
    if not os.path.exists(TEMPDIR):
        os.mkdir(TEMPDIR)
    tempfname = os.path.join(TEMPDIR, 'temp')
    srilm_preprocess_text(raw_text, tempfname)

    # Get perplexity
    outstr = subprocess.check_output('%s -lm %s -write-vocab vocab.lm -debug 3 -ppl %s' %(NGRAM, model_file, tempfname), shell=True)
    print model_file,outstr
    items = outstr.split()
    ppl = 0.0
    for i, item in enumerate(items):
        if item == 'ppl=':
            ppl = float(items[i+1])

    # Cleanup temp file
    os.remove(tempfname)

    return ppl






if __name__ == "__main__":
    pass