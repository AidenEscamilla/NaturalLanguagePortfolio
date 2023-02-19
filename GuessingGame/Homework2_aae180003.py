import sys  # to get the system parameter
import os   # used by pathName
import nltk # used to preprocess and tokenize input
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl  #used to fix certificate error
from random import seed     #used for rng for guessing game
from random import randint

seed(9622)

#this fixes a certificate error I was struggling with for a long time. I had to go through the nltk installer and add the packages so I could use punkt
#After downloading it just prints out a few lines saying punkt is up to date and continues to run the code without the certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



#This function is from class
def pathName(filepath):


    with open(os.path.join(os.getcwd(), filepath), 'r') as f:
        text_in = f.read()
    return text_in

def preprocess_text(inText):
    text = inText.lower()           #Set to lowercase
    tokens_processed = nltk.word_tokenize(text) #tokenize

    #(next two chunks of code are from the class github)
    #This is the preprocessing of words that are only alphabet, not stop words, and length > 5
    tokens_processed = [t for t in tokens_processed if t.isalpha() and          
            len(t) > 5 and
            t not in stopwords.words('english')]
    print('\nNumber of tokens cleaned up: ', len(tokens_processed))

    #get lemmas
    wnl = WordNetLemmatizer()
    lemmas = [wnl.lemmatize(t) for t in tokens_processed]
    #make unique set
    lemmas_unique = list(set(lemmas))


    tags = nltk.pos_tag(lemmas_unique)
    #print('\nPOS Tagged lemmas 0-20 in anat19.txt: ', tags[0:20])

    pos_nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    noun_list = []  
    for pair in tags:               #pair means tuple, for every tuple in tags
        if pair[1] in pos_nouns:    #if pos tag is a noun
            noun_list.append(pair[0])     #add the noun

    print('\nNumber of nouns: ', len(noun_list))

    return tokens_processed, noun_list

#helper function to pick a random word from the top 50 most occuring nouns
def pick_word(word_list):
    return word_list[randint(0,49)]

#guessing game function
def guessing_game(word_list):
    print("\n\nLet's play a word guessing game!")
    print("(To end the game type \'!\' and hit enter)\n")

    player_score = 5        #initialize score
    char_guessed = ''       #initialize char
    
    selected_word = list(pick_word(word_list))  #pick a word
    
    hidden_word = []        #initialize hidden word
    for c in selected_word:
        hidden_word.append('_')


    #print(selected_word)
    ############################################ For Graders! Delete hashtag above to see the first word to help analize how its working

    while player_score >= 0:        #game loop
        index = 0
        print(' '.join(hidden_word))        #correctly format print

        if hidden_word == selected_word:
            print('You solved it!\nCurrent score: ', player_score, '\n\nGuess another word\n')  #format for solving
            
            selected_word = list(pick_word(word_list))              #set up new word
            hidden_word.clear()                 #clean out old hidden word
            for c in selected_word:             #and initialize it again
                hidden_word.append('_')
            
            print(' '.join(hidden_word))          #just like a new game

        char_guessed = input('Guess a letter: ')        #get user input

        if char_guessed == '!':     #choose to exit
            print(":/ i'm sorry i wasn't entertaining enough. Goodbye")
            break
        elif char_guessed in selected_word and char_guessed not in hidden_word:     #if you correctly guess

            for i, c in enumerate(selected_word):
                #go through and add all correct letters
                if char_guessed == c:
                    hidden_word[i] = char_guessed
            #only add 1 score and print 1 time for correct guess
            player_score += 1
            print('Right! Score is ', player_score)
        else:   #else  youre wrong and take away 1
            player_score -= 1
            if player_score < 0:
                print('The word was: ', ''.join(selected_word), '\n')
                print('G A M E   O V E R')
            else:
                print('Sorry, try again. Score is ', player_score)
    


if __name__ == '__main__':
    nltk.download('punkt')

    if len(sys.argv) < 2:
        print('Please enter a filename as a system arg. (Type \'anat19.txt\')')
        quit()
    else:
        print('All good, working now...')
        fp = sys.argv[1]
        text_in = pathName(fp)                   #calls function to open and read file, converts returned string into a string

        tokens, list_of_nouns = preprocess_text(text_in)   #function to clean up text


        unique_Tokens = set(tokens)           #use set to get unique tokens
        lex_Diversity = "{:.2f}".format(len(unique_Tokens)/len(tokens))         #format the lexical diveristy with unique/total
        print('Lexical diversity = ' + lex_Diversity)
        print()

        #edited from class github to add count of noun from non-unique tokens list
        noun_dict = {}
        for t in tokens:
            if t in list_of_nouns:
                if t not in noun_dict:
                    noun_dict[t]= 1
                else:
                    noun_dict[t] += 1

        #sort dictionary and put top 50 into a list
        guessing_words = sorted(noun_dict, key=noun_dict.get, reverse=True)[0:50]

        #print said list and their count
        for i, noun in enumerate(guessing_words):
            print(i+1, '. ', noun, ':', noun_dict[noun])

        #Time to play hehehe
        guessing_game(guessing_words)
        

        quit()