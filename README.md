# NaturalLanguagePortfolio
This is the NLP Portfolio

## NLP Overview pdf

This is an overview of my understanding of Natural Language Processing.
You can read the file [here](Overview_of_NLP.pdf)


## Python Text Processing Homework 1

This program takes in a csv, cleans up the data, corrects format from user, pickles and un-pickles the Person object, and prints the formatted list of info from the csv.

To Run: download the .py and the data file and run in command line with the command 
'python3 Homework1_aae180003.py data/data.csv'.

Python has great strengths in processing with the automatic itterator in for loops. It also has a great regex library that makes it very easy to check text input from a user. It is difficult to implement functions because my ide stops recommending libaray functions outside of main because it isn't certain of parameter types.

Function and comment writing were good reviews for me. Proper implementation of input checking was a nice review. Dynamic variables seem very powerful yet I find myself defining them at the start of most functions anyway. I did learn the scope of python a lot more and that really helped.

You can see the homework [here](TextProcessing/Homework1_aae180003.py)


## Python Guessing Game Homework 2

This program takes in a txt, cleans up the data, calculates the number of tokens, unique tokens, and lexical diversity in such txt file.

To Run: download the .py and the data file and run in command line with the command 
'python3 Homework2_aae180003.py anat19.txt'.

After cleaning up the input text and getting a set of unique lemmas the program uses nltk to categorize the parts of speech. The program then calculates and prints the top 50 nouns and enters into a word guessing game. It's basically a customized hangman game.(you might have to download certain nltk packages to run the code)

You can see the homework [here](GuessingGame/Homework2_aae180003.py)

## Python WordNet Homework 3

This program runs through the use and practice of WordNet, SentiWordNet, and collocations.

After picking my noun and verbs I go through the WordNet hierarchy and type out plenty of summaries and reflections in the google collab. Next is the similarity between words which uses SentiWordNet. It seems very primative with only 3 scores that usually come out to whole fractions. But it's fun to see what words it thinks are positive or negative, and what words it thinks are and aren't similar. Last, collocations seems to just be a quick and easy way to find the bigrams in a text. I know it's trying to find collections that are greater than the sum of their parts but my results seem lacking.

You can see the Collab [here](WordNet/WordNetHw3.pdf)

## Python N-Grams Homework 4

There are two programs here. One that reads in the training data and pickles the dictionaries created, and another that opens the pickled dictionaries and tried to classify the language of a sentence given a test data set.

The programs only use unigrams and bigrams but the scalability is visable to higher n-grams. Laplace smoothing is used to get a more accurate prediction. Accuracy from LangId.test is at 96%.

To Run: download the .py files and the data file and run in command line with the command 

'python3 Homework4_createNGrams_aae180003.py data/LangId.train.English data/LangId.train.French data/LangId.train.Italian'.

then:

'python3 Homework4_languageDetection_aae180003.py data/LangId.test data/LangId.sol'

(Your machine might ask you for permission to write because the language detection writes a file as output)


You can see the training [here](nGrams/Homework4_createNGrams_aae180003.py)
You can see the language detection [here](nGrams/Homework4_languageDetection_aae180003.py)
And the overview/reflection write up [here](nGrams/n-grams_WriteUp.pdf)

## Parsing Overview pdf

This is an overview of my understanding of three different parsing styles.
Constituent, dependency, and SRL parsing. 

Constituent seems a lot more intuitive since I learned POS tagging and it follows that hierarchy closely.
Dependency seems very helpful for visual people but not so much myself. It tagged a weird part of my example sentence as a conjunction .
SRL in nicknamed shallow parsing for a reason. It separates the verbs into different predicates and gives them all their proper argument assignments. It seemed fitting for the sentence but uses too many broad categories for my liking.

You can read the file [here](Parsing/ParsingByHand.pdf)

## WebCrawler

Here we were supposed to make a web crawler that scraped text from websites about a certain topic. I chose genius.com to collect song lyric text. I started with the url for a single artist and crawled the links to every one of their songs. From there I gathered the lyrics, cleaned them up, and used nltk to get the frequency and importance of words in every song.

The importance of every word is calculated with term frequency in a document (tf) and inverse document frequency (idf) with the formula tf*idf. This allows the program to print a list of the top 25 words on every page. With a more specific topic this would show a lot of overlapping terms but because most of the songs are different the terms don't overlap for me.

Do be careful trying to run this code because it creates a file for ever song it finds as per part of the homework requirments.

See the code [here](WebCrawler/Homework5_aae180003.py)
You can read the find and example chatbot dialogue [here](WebCrawler/Webcrawler_Findings.pdf)


## Text Classification

Here we were trying different methods at classification. From Naive Bayes, to logistical regression, to neural network. I used a data set where someone tried to categorize poems into 4 categories. Those classifications being death, affection, environment, music. I couldn't get much higher than 40% accuracy for Naive Bayes but I am impressed it even got that far. Poems are very open for interpretation and it seems like the classification prediction got lost in the bag of words approach. Preserving the word order might prove to be more accurate.

See the colab notebook [here](TextClassification/TextClassification_ipynb.pdf)


## Chat bot

This my attempt at a utility chat bot similar to a bot that is only good at one thing like helping with an application. It starts with easy recommendations like random songs from the database or lets you choose a category that i've created arbitrarily. Hopefully I can use topic modeling from class to create user specific topics/Categories and use that Category function to choose from those predicted category labels.
The 'meat and potatoes' of the bot is the last function that recommends the user songs based on their own spotify. After collecting every song the user follows and adding that to an SQLite database, the program webcrawls genius.com for song lyrics, creates a tf_idf vector space, and returns the top 10 songs closest in the vector space. 
It takes a bit of time to go through all those steps but after the first menu loop files are pickled in order to greatly improve performance. If the user has a lot of songs, creating the database by webcrawling takes a long long time. But, after the first run through you don't need to create the database again unless there's a new song you want to run through the program.The song recommendations come from a Pandas data frame the holds data from the users spotify, and the spotify million songs .csv file.
(Note, i'm pretty sure if you want to run the code yourself you have to let me know your spotify email. Also there's a private key needed as an environment variable that I need to figure out how/if i can share that without giving away the info.)

See the code [here](Chatbot/RecommendingRobot.py)
*Don't forget to download the requirements
