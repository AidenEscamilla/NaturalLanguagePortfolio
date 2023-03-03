import sys  # to get the system parameter
import os   # used by pathName
import pickle #yknow to pickle the files because building ngrams every time you want to call the is a lot. Thx python generator function
import nltk # used to preprocess and tokenize input
from nltk.util import ngrams


#This function creates and returns the unigram and bigram dictionaries
def getNGrams(filepath):
    text_in = str()

    with open(os.path.join(os.getcwd(), filepath), 'r') as f:
        text_in = f.read()
    
    text_in = text_in.replace("\n" and ',' and '\'', " ")
    tokens = nltk.word_tokenize(text_in)
    unigrams = ngrams(tokens, 1)
    bigrams = ngrams(tokens, 2)

    #from class github
    unigram_dict = {}
    for unigram in set(unigrams):
        unigram_dict[unigram[0]] = text_in.count(unigram[0])

    bigram_dict = {}
    for bigram in set(bigrams):
        if bigram not in bigram_dict:
            bi = bigram[0] + ' ' + bigram[1]
            bigram_dict[bi] = text_in.count(bi)

        
    return unigram_dict, bigram_dict



if __name__ == '__main__':
    #nltk.download('punkt')
    dictionaries = []

    if len(sys.argv) < 4:
        print('Please enter three filename as a system arg. (Type \'data/file1_name_here\' \'data/file2_name_here\' \'data/file3_name_here\' )')
        quit()
    else:
        print('All good, working now...')
    
    fp = sys.argv[1]
    dictionaries = getNGrams(fp)                   #calls function to open and read file, returns the two dictionaries for english

    with open('unigrams_Eng.pickle', 'wb') as handle:
        pickle.dump(dictionaries[0], handle)
    
    with open('bigrams_Eng.pickle', 'wb') as handle:
        pickle.dump(dictionaries[1], handle)



    fp = sys.argv[2]
    dictionaries = getNGrams(fp)                   #calls function to open and read file, returns the two dictionaries for French

    with open('unigrams_Fren.pickle', 'wb') as handle:
        pickle.dump(dictionaries[0], handle)
    
    with open('bigrams_Fren.pickle', 'wb') as handle:
        pickle.dump(dictionaries[1], handle)



    fp = sys.argv[3]
    dictionaries = getNGrams(fp)                   #calls function to open and read file, returns the two dictionaries for Italian

    with open('unigrams_Ital.pickle', 'wb') as handle:
        pickle.dump(dictionaries[0], handle)
    
    with open('bigrams_Ital.pickle', 'wb') as handle:
        pickle.dump(dictionaries[1], handle)

    quit()
