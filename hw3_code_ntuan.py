from nltk import word_tokenize, sent_tokenize
from os import listdir, system, popen
from os.path import isfile, join, basename
from itertools import chain
import numpy as np
from math import log
from sklearn.svm import SVC
import itertools
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from collections import Counter, defaultdict
from nltk.tree import Tree
import pdb
import re

# ntuan_home = "/Users/admin/Documents/cis530/hw3/data/train/"
ntuan_home = "/home1/n/ntuan/cis530/hw3/data/train/"
ntuan_home_2 = "/home1/n/ntuan/cis530/hw3/"
ntuan_numbers = re.compile(r'(\d+)')

#Credit: http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
def flatten(listoflists):
    return list(itertools.chain.from_iterable(listoflists))

def numericalSort(value):
    parts = ntuan_numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def standardize(rawexcerpt):
    return [w.encode('utf8') for w in word_tokenize(rawexcerpt.decode('utf8').lower())]

def load_file_excerpts(filepath):
    excerpts = []
    with open(filepath, 'rU') as fin:
        for line in fin:
            excerpts.append(standardize(line.strip()))
    return excerpts

def get_all_files(directory):
    return sorted([join(directory, f) for f in listdir(directory) if isfile(join(directory, f))], key = numericalSort)

def write_to_file(list_of_stuff, file_name):
    #write each thing in list_of_stuff to one line in <file_name>.txt
    if not isfile(join(ntuan_home_2, file_name)):
        with open(ntuan_home_2 + file_name, 'w') as csvfile:
            csvfile.writelines(str(item_i) + '\n' for item_i in list_of_stuff)

def run_Stanford_coreNLP():
    train_dir = ntuan_home_2 + "data/train/"
    test_dir = ntuan_home_2 + "data/test/"
    train_files = get_all_files(train_dir)
    test_files = get_all_files(test_dir)
    write_to_file(train_files, "train_file_list.txt")
    write_to_file(test_files, "test_file_list.txt")
    file_list = [ntuan_home_2 + "train_file_list.txt", ntuan_home_2 + "test_file_list.txt"]
    for file_i in file_list:
        system("cd /home1/n/ntuan/cis530/hw3/stanford-corenlp-2012-07-09/;\
        java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:\
        xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP \
        -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist "
          + file_i + " -outputDirectory " + ntuan_home_2)

def extract_pos_tags(xml_directory):
#returns all the unique pos tags from all documents in xml_directory.
    xml = get_all_files(xml_directory)
    tags = []
    for file_i in xml:
        curr_file = open(file_i)
        tags += [line.strip().split("POS>")[1][:-2] for line in curr_file.read().split('\n') if '</POS>' in line]
        curr_file.close()
    tags = sorted(list(set(tags)))
    write_to_file(tags, "hw3_4-1.txt")
    return tags

def map_pos_tags(xml_filename, pos_tag_list):
#takes an xml file path and the list of known POS tags (output from 4.1) as
#input and returns a vector in the feature space of the known POS tag list.
    curr_file = open(xml_filename)
    tags = [line.strip().split("POS>")[1][:-2] for line in curr_file.read().split('\n') if '</POS>' in line]
    curr_file.close()
    count = Counter(tags)
    num_tok = len(tags)
    num_tag = len(pos_tag_list)
    pos_tags_vec = [0]*num_tag
    return [float(count[pos_tag_list[i]])/num_tok for i in range(num_tag)]

def map_universal_tags(ptb_pos_feat_vector, pos_tag_list, ptb_google_mapping, universal_tag_list):
#The function returns a vector in the feature space of the universal tag list.
#It returns a list of integers with the same size as universal tag list.
#Each element in the returned list is equal to the fraction of tokens in the text
#(represented by the input vector) with the corresponding universal POS tag
#in universal tag list.
    ggl_vec = [0]*len(universal_tag_list)
    N = len(ptb_pos_feat_vector)
    for i in range(N):
        pos_tag = pos_tag_list[i]
        ggl_tag = ptb_google_mapping[pos_tag]
        index = universal_tag_list.index(ggl_tag)
        ggl_vec[index] = ptb_pos_feat_vector[i]
    return ggl_vec

def extract_ner_tags(xml_directory):
    xml = get_all_files(xml_directory)
    tags = []
    for file_i in xml:
        curr_file = open(file_i)
        tags += [line.strip().split("NER>")[1][:-2] for line in curr_file.read().split('\n') if "</NER>" in line]
        curr_file.close()
    tags = sorted(list(set(tags)))
    write_to_file(tags, "hw3_5-1.txt")
    return tags

def map_named_entity_tags(xml_filename, entity_list):
#takes an xml file path and the list of named entity classes (output from 5.1)
#as input and returns a vector in the feature space of the Named Entity list.
#It returns a list of real numbers with the same size as the entity list.
#Each element in the returned list is equal to the number of times the
#corresponding named entity class in entity list occurred in the xml input file,
#divided by the number of tokens in the input file.
    curr_file = open(xml_filename)
    tags = [line.strip().split("NER>")[1][:-2] for line in curr_file.read().split('\n') if "</NER>" in line]
    curr_file.close()
    count = Counter(tags)
    num_tok = len(tags)
    num_tag = len(entity_list)
    pos_tags_vec = [0]*num_tag
    return [float(count[entity_list[i]])/num_tok for i in range(num_tag)]

def extract_dependencies(xml_directory):
    xml = get_all_files(xml_directory)
    read_flag = False
    dep_tag = []
    for file_i in xml:
        curr_file = open(file_i)
        for line in curr_file.read().split("\n"):
            line = line.strip()
            if read_flag and "dep type" in line:
                dep_tag.append(line.split('"')[-2])
            if line == "</basic-dependencies>":
                read_flag = False
            if line == "<basic-dependencies>":
                read_flag = True
        curr_file.close()
        read_flag = False
    dep_tag = sorted(list(set(dep_tag)))
    write_to_file(dep_tag, "hw3_6-1.txt")
    return dep_tag

def map_dependencies(xml_filename, dependency_list):
#takes an xml file path and the list of dependency types (output in 6.1)
#as input and returns a list of the same length as dependency list.
#Each element in the output list takes the value of the number of times
#the corresponding dependency in dependency list appeared in the xml input file
#normalized by the number of all dependencies in the text.
    curr_file = open(xml_filename)
    read_flag = False
    dep_tag = []
    for line in curr_file.read().split("\n"):
        line = line.strip()
        if read_flag and "dep type" in line:
            dep_tag.append(line.split('"')[-2])
        if line == "</basic-dependencies>":
            read_flag = False
        if line == "<basic-dependencies>":
            read_flag = True
    curr_file.close()
    N = len(dep_tag)
    count = Counter(dep_tag)
    if not N:
        return [0]*len(dependency_list)
    return [float(count[tag])/N for tag in dependency_list]

def extract_prod_rules(xml_directory):
    xml = get_all_files(xml_directory)
    tree_str = []
    for file_i in xml:
        curr_file = open(file_i)
        tree_str += [line.split("<parse>")[-1].split("<")[0].strip() for line in curr_file.read().split('\n') if ("<parse>" in line)]
        curr_file.close()
    CFG_trees = [Tree.fromstring(rule) for rule in set(tree_str)]
    parse = sorted(list(set(["_".join(rule.__str__().replace('->',' ').split()) for t in CFG_trees for rule in t.productions() if "'" not in rule.__str__()])))
    write_to_file(parse, "hw3_7-1.txt")
    return parse

def map_prod_rules(xml_filename, rules_list):
#takes an xml file and a list of rules in the format specified above and returns
#a list of the same length as rules list. The element in the list takes the
#value 1 if its corresponding rule in rules list appeared in the xml input file,
#0 otherwise.
    tree_str = []
    curr_file = open(xml_filename)
    tree_str = [line.split("<parse>")[-1].split("<")[0].strip() for line in curr_file.read().split('\n') if ("<parse>" in line)]
    CFG_trees = [Tree.fromstring(rule) for rule in set(tree_str)]
    parse = ["_".join(rule.__str__().replace('->',' ').split()) for t in CFG_trees for rule in t.productions() if not ("'" in rule.__str__() and len(rule.__str__().split("->")) == 2)]
    parse = set(parse)
    curr_file.close()
    return [int(rule in parse) for rule in rules_list]

# def generate_cluster_codes(brown_file_path):
#takes the brown cluster file path as input and returns a list of unique cluster
#names/codes present in the file. Append a code 8888 to the end of this list for
#words in your test data that are not present in the precomputed Brown clusters.

def generate_word_cluster_mapping(brown_file_path):
#takes the path to the brown cluster file (str) as input and returns a dict
#containing a mapping from words occurring in the brown cluster files to
#their cluster codes.
    word_code_map = {}
    code_list = set()
    curr_file = open("/project/cis/nlp/data/corpora/MetaOptimize/BrownWC/brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt")
    for line in curr_file.read().strip().split('\n'):
        things = line.split()
        code = things[0]
        word = things[1]
        code_list.add(code)
        word_code_map[word] = code
    curr_file.close()
    return sorted(list(code_list)) + ["8888"], word_code_map

def map_brown_clusters(xml_file_path, cluster_code_list, word_cluster_mapping):
#produces a representation that reflects the normalized frequency of each known
#Brown cluster in the given text. For words that do not appear in the
#precomputed Brown clusters, their code should be 8888. Param: cluster code list
#(list) and word cluster mapping (dict) are outputs of the helper functions below..
#Return: A vector (or list) of the same length of cluster code list.
#Each element in the output list takes the value of the number of times
#the corresponding cluster in cluster code list appeared in the given text
#divided by the number of all words in the text.
    curr_file = open(xml_file_path)
    words = [line.split("word>")[-2][:-2] for line in curr_file.read().split('\n') if ("<word>" in line)]
    N = len(words)
    curr_file.close()
    code_list = [word_cluster_mapping[word_i] if word_i in word_cluster_mapping else "8888" for word_i in words]
    code_list = Counter(code_list)
    return [float(code_list[code_i])/N for code_i in cluster_code_list]

def createPOSFeat(xml_dir, pos_tag_list):
    xml = get_all_files(xml_dir)
    return [map_pos_tags(file_i, pos_tag_list) for file_i in xml]

def createUniversalPOSFeat(pos_feat_2D_array, pos_tag_list, ptb_google_mapping, universal_tag_list):
    return [map_universal_tags(ptb_pos_feat_vector, pos_tag_list, ptb_google_mapping, universal_tag_list) for ptb_pos_feat_vector in pos_feat_2D_array]

def createNERFeat(xml_dir, entity_list):
    xml = get_all_files(xml_dir)
    return [map_named_entity_tags(file_i, entity_list) for file_i in xml]

def createDependencyFeat(xml_dir, dependency_list):
    xml = get_all_files(xml_dir)
    return [map_dependencies(file_i, dependency_list) for file_i in xml]

def createSyntaticProductionFeat(xml_dir, rules_list):
    xml = get_all_files(xml_dir)
    return [map_prod_rules(file_i, rules_list) for file_i in xml]

def createBrownClusterFeat(xml_dir, cluster_code_list, word_cluster_mapping):
    xml = get_all_files(xml_dir)
    return [map_brown_clusters(file_i, cluster_code_list, word_cluster_mapping) for file_i in xml]

def run_classifier(X_train, y_train, X_test, predicted_labels_file):
    clf = SVC()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    write_to_file(pred, predicted_labels_file)
    return pred

def SVC_opt(X_train, y_train, X_test, p_flag = True):
    result = []
    params = []
    max_acc = 0

    for (C, gamma, kernel, degree, coef0, tol) in itertools.product([0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 'auto'],
                                                                ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                                                                [2, 3],
                                                                [0.0, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
        if kernel != 'poly' and degree > 2:
            continue

        if kernel not in ['rbf', 'poly', 'sigmoid'] and gamma != 'auto':
            continue

        if kernel not in ['poly', 'sigmoid'] and coef0 != 0.0:
            continue

        clf = make_pipeline(StandardScaler(),
                            SVC(C=C,
                                gamma=gamma,
                                kernel=kernel,
                                degree=degree,
                                coef0=coef0,
                                tol=tol))
        cv_scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, n_jobs=-1)
        accuracy = np.mean(cv_scores)
        if accuracy > max_acc:
            max_acc = accuracy
            params = [C, gamma, kernel, degree, coef0, tol, accuracy]
        if p_flag:
            print(C, gamma, kernel, degree, coef0, tol, accuracy)
    clf = SVC(C = params[0], gamma = params[1], kernel = params[2], degree = params[3], coef0 = params[4], tol = params[5])
    pred = clf.predict(X_test)
    write_to_file(pred, "best_model.txt")
    write_to_file(params, "best_params.txt")
    print(max_acc)
    return max_acc, params

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

def compute_read_features(line):
    sents = [w.encode('utf8') for w in sent_tokenize(line.decode('utf8'))]
    total_sent_length = sum([len(sent) for sent in sents])
    num_sents = len(sents)
    tokens = flatten([standardize(sent) for sent in sents])
    N = len(tokens)
    counts = Counter(tokens)
    type_tok = float(len(counts))/(N)
    unigram = {word:float(counts[word])/N for word in counts}
    #count all tokens that are not punctuations in cloze abstract
    words = [word for word in tokens if not len(word) == 1 or word.isalpha()]
    extra_long = len([word for word in words if len(word) > 13])
    word_len = np.array([len(word) for word in words])
    avg_word_len = np.mean(word_len)
    median_word_len = np.median(word_len)
    num_words = len(words)
    num_syl = sum([nsyl(word.lower()) for word in words])
    avg_syl_per_word = float(num_syl)/num_words
    avg_word_per_sent = float(num_words)/num_sents
    FL_score = 206.835 - 1.015*avg_word_per_sent - 84.6*avg_syl_per_word
    FK_score = 0.39*avg_word_per_sent + 11.8*avg_syl_per_word - 15.59
    entropy = -sum([unigram[word_i]*log(unigram[word_i],2) for word_i in unigram])
    return [FK_score, FL_score, entropy, type_tok, avg_word_len, median_word_len, extra_long]

def readability_features(xml_dir):
    xml = get_all_files(xml_dir)
    feature_vec = []
    for file_i in xml:
        curr_file = open(file_i)
        corpus = curr_file.read().strip()
        feature_vec.append(compute_read_features(corpus))
    return feature_vec

if __name__ == "__main__":

    curr_file = open("/home1/n/ntuan/cis530/hw3/data/en-ptb.map")
    ptb_google_mapping = {}
    for line in curr_file.read().strip().split("\n"):
        ptb_google_mapping[line.split()[0]] = line.split()[1]
    curr_file.close()
    universal_tag_list = sorted(list(ptb_google_mapping.values()))

    #run_Stanford_coreNLP()

    pos_tags_train = extract_pos_tags("/home1/n/ntuan/cis530/hw3/train_parse/")

    pos_feat_2D_array_train = createPOSFeat("/home1/n/ntuan/cis530/hw3/train_parse/", pos_tags_train)
    UniversalPOSFeat_train = np.asarray(createUniversalPOSFeat(pos_feat_2D_array_train, pos_tags_train, ptb_google_mapping, universal_tag_list))
    pos_feat_2D_array_train = np.asarray(pos_feat_2D_array_train)

    pos_feat_2D_array_test = createPOSFeat("/home1/n/ntuan/cis530/hw3/test_parse/", pos_tags_train)
    UniversalPOSFeat_test = np.asarray(createUniversalPOSFeat(pos_feat_2D_array_test, pos_tags_train, ptb_google_mapping, universal_tag_list))
    pos_feat_2D_array_test = np.asarray(pos_feat_2D_array_test)

    deps_train = extract_dependencies("/home1/n/ntuan/cis530/hw3/train_parse/")
    DependencyFeat_train = np.asarray(createDependencyFeat("/home1/n/ntuan/cis530/hw3/train_parse/", deps_train))
    DependencyFeat_test = np.asarray(createDependencyFeat("/home1/n/ntuan/cis530/hw3/test_parse/", deps_train))

    ner_tags_train = extract_ner_tags("/home1/n/ntuan/cis530/hw3/train_parse/")
    NERFeat_train = np.asarray(createNERFeat("/home1/n/ntuan/cis530/hw3/train_parse/", ner_tags_train))
    NERFeat_test = np.asarray(createNERFeat("/home1/n/ntuan/cis530/hw3/test_parse/", ner_tags_train))

    prod_rules_train = extract_prod_rules("/home1/n/ntuan/cis530/hw3/train_parse/")
    SyntaticProductionFeat_train = np.asarray(createSyntaticProductionFeat("/home1/n/ntuan/cis530/hw3/train_parse/", prod_rules_train))
    SyntaticProductionFeat_test = np.asarray(createSyntaticProductionFeat("/home1/n/ntuan/cis530/hw3/test_parse/", prod_rules_train))

    cluster_code_list, word_cluster_mapping = generate_word_cluster_mapping("/project/cis/nlp/data/corpora/MetaOptimize/BrownWC/brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt")
    BrownClusterFeat_train = np.asarray(createBrownClusterFeat("/home1/n/ntuan/cis530/hw3/train_parse/", cluster_code_list, word_cluster_mapping))
    BrownClusterFeat_test = np.asarray(createBrownClusterFeat("/home1/n/ntuan/cis530/hw3/test_parse/", cluster_code_list, word_cluster_mapping))

    readability_features_test = np.asarray(readability_features("/home1/n/ntuan/cis530/hw3/data/test/"))
    readability_features_train = np.asarray(readability_features("/home1/n/ntuan/cis530/hw3/data/train/"))

    X_train = np.hstack((UniversalPOSFeat_train, pos_feat_2D_array_train, DependencyFeat_train, NERFeat_train, SyntaticProductionFeat_train, BrownClusterFeat_train, readability_features_train))
    X_test = np.hstack((UniversalPOSFeat_test, pos_feat_2D_array_test, DependencyFeat_test, NERFeat_test, SyntaticProductionFeat_test, BrownClusterFeat_test, readability_features_test))
    print(X_train.shape)
    print(X_test.shape)
    np.savetxt("X_train.txt", X_train)
    np.savetxt("X_test.txt", X_test)
    y_train = [int(file_i.split("/")[-1].split("_")[0] == "gina") for file_i in get_all_files(ntuan_home)]

    # run_classifier(X_train, y_train, X_test, "modelscore.txt")
    SVC_opt(X_train, y_train, X_test)
