import pickle #yknow to pickle the files because building ngrams every time you want to call the is a lot. Thx python generator function
import sys  # to get the system parameter
import os   # used by pathName
import nltk # used to preprocess and tokenize input
from nltk import word_tokenize  #To get tokens
import math #for calculating probabilities
from nltk.util import ngrams


#This function is from class
def OpenFile(filepath):


    with open(os.path.join(os.getcwd(), filepath), 'r') as f:
        text_in = f.read()
    return text_in



def compute_prob(text, unigram_dict, bigram_dict, V):
    # V is the vocabulary size in the training data (unique tokens)
    
    unigrams_test = word_tokenize(text)
    bigrams_test = list(ngrams(unigrams_test, 2))
    p_laplace = 0  # calculate p using Laplace smoothing
    #Math of Laplace smoothin below is:  (b + 1) / (u + v) where b is the bigram count, u is the unigram count of the first word in the bigram, and v is the total vocabulary size
    #From class github
    for bigram in bigrams_test:
        b = bigram_dict[bigram] if bigram in bigram_dict else 0
        u = unigram_dict[bigram[0]] if bigram[0] in unigram_dict else 0
        p_laplace = p_laplace + abs(math.log((b + 1) / (u + V), 10))
    return  p_laplace/1000




if __name__ == '__main__':
    

    if len(sys.argv) < 3:
        print('Please enter a filename as a system arg. (Type \'data/LangId.test\' \'data/LangId.sol\' )')
        quit()
    else:
        print('All good, working now...')

    fp = sys.argv[1]
    test_data = OpenFile(fp).replace('\xad', '').replace('\'', " ").splitlines()       #calls function to open and read file, returns the testing set without soft hyphen character,  and splits each line for language identification
        #replacing the apostrophes got me 3 more correct classifications


    fp = sys.argv[2]
    solutions = OpenFile(fp).splitlines()               #calls function to open solution classificatinos and split by new lines
    classifications = []

    #fill list with classifications
    for line in solutions:
        classifications.append(line.split()[1])
    

    #Open all 6 pickled dictionaries
    with open('unigrams_Eng.pickle', 'rb') as handle:
        E_uni_dict = pickle.load(handle)
    
    with open('bigrams_Eng.pickle', 'rb') as handle:
        E_bi_dict = pickle.load(handle)
    
    
    with open('unigrams_Fren.pickle', 'rb') as handle:
        F_uni_dict = pickle.load(handle)
    
    with open('bigrams_Fren.pickle', 'rb') as handle:
        F_bi_dict = pickle.load(handle)


    with open('unigrams_Ital.pickle', 'rb') as handle:
        I_uni_dict = pickle.load(handle)
    
    with open('bigrams_Ital.pickle', 'rb') as handle:
        I_bi_dict = pickle.load(handle)

    vocab = len(E_uni_dict) + len(F_uni_dict) + len(I_uni_dict)


    #setup output file
    output_file = open("LangIDOutput.txt", "w")


    #Get vocab size of entire training set
    vocab = len(E_uni_dict) + len(F_uni_dict) + len(I_uni_dict)

    
    #write all probabilities for all lines to file
    for test_line in test_data:
        p_eng = compute_prob(test_line, E_uni_dict, E_bi_dict, vocab)
        p_fre = compute_prob(test_line, F_uni_dict, F_bi_dict, vocab)
        p_ita = compute_prob(test_line, I_uni_dict, I_bi_dict, vocab)


        maxLanguage = 'English'
        languageProb = "{:.4f}".format(p_eng)

        if p_fre > p_eng and p_fre > p_ita:
            maxLanguage = 'French'
            languageProb = "{:.4}".format(p_fre)
        elif p_ita > p_fre and p_ita > p_eng :
            maxLanguage = 'Italian'
            languageProb = "{:.4f}".format(p_ita)

        output_file.write(maxLanguage + ' ' + languageProb + "\n")
    
    output_file.close()

   
   
    correct_prediction = 0
    predictions = OpenFile('LangIDOutput.txt').splitlines()
    
    #Used to debug and understand
    #wrong = [43, 83, 108, 113, 156, 163, 168, 193, 205, 245, 250, 280]
    wrong = []

    #Used to debug and understand
    #for line in wrong:
    #    print('Eng')
    #    p_eng = compute_prob(test_data[line], E_uni_dict, E_bi_dict, vocab)
    #    print('Fre')
    #    p_fre = compute_prob(test_data[line], F_uni_dict, F_bi_dict, vocab)
    #    print('Ita')
    #    p_ita = compute_prob(test_data[line], I_uni_dict, I_bi_dict, vocab)
    #quit()

    for i, classification in enumerate(classifications):
        if predictions[i].split()[0] == classification:
            correct_prediction += 1
        else:
            wrong.append(i)
            print('Line ', i, " incorrect\nPredicition: ", predictions[i].split()[0], " Actual Language: ", classification, '\n')
    
    print('correct correct: ', correct_prediction)
    print('percentage: ', correct_prediction/300)

    
    #Used to debug and understand
    #for i in wrong:
    #    p_eng = compute_prob(test_line, E_uni_dict, E_bi_dict, vocab)
    #    p_fre = compute_prob(test_line, F_uni_dict, F_bi_dict, vocab)
    #    p_ita = compute_prob(test_line, I_uni_dict, I_bi_dict, vocab)
    #    print(i, ' E: ', p_eng, "F: ",  p_fre, 'I: ', p_ita, "\n")
    

    quit()