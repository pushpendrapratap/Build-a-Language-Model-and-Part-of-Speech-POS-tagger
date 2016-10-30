import math
import nltk
import time

import numpy as np         #  courtesy Pushpendra pratap
import collections 

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    #############################################################################  courtesy Pushpendra pratap
    # nested_tokens_list = [nltk.word_tokenize(i) for i in training_corpus]
    nested_tokens_list = [i.split() for i in training_corpus]
    for i in nested_tokens_list:                                # since, unigrams contain only START SYMBOL
        # i.insert(0,START_SYMBOL)           # or, i[0:0]=START_SYMBOL
        i.append(STOP_SYMBOL)  

    unigram_nested_tokens = [j for i in nested_tokens_list for j in i]     
    length_of_unigram_corpus = len(unigram_nested_tokens)
    
    temp_unigram_p = collections.Counter(unigram_nested_tokens)
    for i in temp_unigram_p:
        unigram_p[i] = math.log(temp_unigram_p[i]/(1.0*length_of_unigram_corpus), 2)




    START_nested_tokens_list = nested_tokens_list # since, bigrams and trigrams will contain both START and STOP SYMBOL
    for i in START_nested_tokens_list:
        i.insert(0,START_SYMBOL)           # or, i[0:0]=START_SYMBOL 

    START_unigram_nested_tokens = [j for i in START_nested_tokens_list for j in i] 

    bigram_nested_tokens_temp = list(list(nltk.bigrams(i)) for i in START_nested_tokens_list)
    bigram_nested_tokens = [j for i in bigram_nested_tokens_temp for j in i]

    # trigram_nested_tokens = list(nltk.trigrams(START_unigram_nested_tokens))

    START_temp_unigram_p = collections.Counter(START_unigram_nested_tokens)

    temp_bigram_p = collections.Counter(bigram_nested_tokens)
    for i in temp_bigram_p:
        bigram_p[i] = math.log(temp_bigram_p[i]/(1.0*START_temp_unigram_p[i[0]]), 2)




    DOUBLE_START_nested_tokens_list = START_nested_tokens_list # since, bigrams and trigrams will contain both START and STOP SYMBOL
    for i in DOUBLE_START_nested_tokens_list:                  # trigrams conatin two START_SYMBOL
        i.insert(0,START_SYMBOL)           # or, i[0:0]=START_SYMBOL 

    DOUBLE_START_unigram_nested_tokens = [j for i in DOUBLE_START_nested_tokens_list for j in i] 

    DOUBLE_bigram_nested_tokens_temp = list(list(nltk.bigrams(i)) for i in DOUBLE_START_nested_tokens_list)
    DOUBLE_bigram_nested_tokens = [j for i in DOUBLE_bigram_nested_tokens_temp for j in i] 

    trigram_nested_tokens_temp = list(list(nltk.trigrams(i)) for i in DOUBLE_START_nested_tokens_list) 
    trigram_nested_tokens = [j for i in trigram_nested_tokens_temp for j in i] 

    DOUBLE_temp_bigram_p = collections.Counter(DOUBLE_bigram_nested_tokens)
    temp_trigram_p = collections.Counter(trigram_nested_tokens)
    for i in temp_trigram_p:
        temp_tuple = (i[0],i[1])
        trigram_p[i] = math.log(temp_trigram_p[i]/(1.0*DOUBLE_temp_bigram_p[temp_tuple]), 2)
    ##############################################################################

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        # outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    ###############################################################       courtesy Pushpendra pratap
    new_nested_tokens_list = [i.split() for i in corpus]

    for i in new_nested_tokens_list:
        i.append(STOP_SYMBOL)
 
    if(n>=2):
        for i in new_nested_tokens_list:
            i.insert(0,START_SYMBOL)
            if(n==3):
                i.insert(0,START_SYMBOL)

    general_tokens_list = new_nested_tokens_list

    if(n==2):
        general_tokens_list = list(list(nltk.bigrams(i)) for i in general_tokens_list) 
    elif(n==3):
        general_tokens_list = list(list(nltk.trigrams(i)) for i in general_tokens_list)   

    for j in range(len(new_nested_tokens_list)):
        temp = 0.0
        for k in general_tokens_list[j]:
            x = ngram_p.get(k, MINUS_INFINITY_SENTENCE_LOG_PROB)
            if (x==MINUS_INFINITY_SENTENCE_LOG_PROB):
                temp = x
                break 
            else:
                temp = temp + x
        scores.append(temp)
    ###############################################################

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores

def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    ######################################################################## Courtesy pushpendra pratap
    # lambda_list = [0.3333333333333333, 0.3333333333333333, 0.3333333333333334]  # upto 16 decimal digits 
    x = 1/3.0
    lambda_list = [x, x, 1-x-x]

    # new_nested_tokens_list = [nltk.word_tokenize(i) for i in corpus]
    new_nested_tokens_list = [i.split() for i in corpus]

    for i in new_nested_tokens_list:
        i.insert(0,START_SYMBOL)
        i.insert(0,START_SYMBOL)
        i.append(STOP_SYMBOL)

    new_trigram_nested_tokens = list(list(nltk.trigrams(i)) for i in new_nested_tokens_list)

    for j in range(len(new_nested_tokens_list)):
        temp = 0.0
        for k in new_trigram_nested_tokens[j]:  
            temp = temp + math.log( ((lambda_list[0] * math.pow(2,unigrams.get(k[2], MINUS_INFINITY_SENTENCE_LOG_PROB))) + \
                            (lambda_list[1] * math.pow(2,bigrams.get((k[1],k[2]), MINUS_INFINITY_SENTENCE_LOG_PROB))) + \
                            (lambda_list[2] * math.pow(2,trigrams.get(k, MINUS_INFINITY_SENTENCE_LOG_PROB)))),2 )  

        scores.append(temp)
    #########################################################################

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
