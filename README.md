# NaturalLanguagePortfolio
This is the NLP Portfolio

## NLP Overview pdf

This is an overview of my understanding of Natural Language Processing.
You can read the file [here](Overview_of_NLP.pdf)


## Python Text Processing Homework 1

This program takes in a csv, cleans up the data, corrects format from user, pickles and un-pickles the Person object, and prints the formatted list of info from the csv.

To Run: download the .py and the data file and run in command line with the command 
'python Homework1_aae180003.py data/data.csv'.

Python has great strengths in processing with the automatic itterator in for loops. It also has a great regex library that makes it very easy to check text input from a user. It is difficult to implement functions because my ide stops recommending libaray functions outside of main because it isn't certain of parameter types.

Function and comment writing were good reviews for me. Proper implementation of input checking was a nice review. Dynamic variables seem very powerful yet I find myself defining them at the start of most functions anyway. I did learn the scope of python a lot more and that really helped.

You can see the homework [here](TextProcessing/Homework1_aae180003.py)


## Python Guessing Game Homework 2

This program takes in a txt, cleans up the data, calculates the number of tokens, unique tokens, and lexical diversity in such txt file.

To Run: download the .py and the data file and run in command line with the command 
'python Homework2_aae180003.py anat19.txt'.

After cleaning up the input text and getting a set of unique lemmas the program uses nltk to categorize the parts of speech. The program then calculates and prints the top 50 nouns and enters into a word guessing game. It's basically a customized hangman game.(you might have to download certain nltk packages to run the code)

You can see the homework [here](GuessingGame/Homework2_aae180003.py)

## Python WordNet Homework 3

This program runs through the use and practice of WordNet, SentiWordNet, and collocations.

After picking my noun and verbs I go through the WordNet hierarchy and type out plenty of summaries and reflections in the google collab. Next is the similarity between words which uses SentiWordNet. It seems very primative with only 3 scores that usually come out to whole fractions. But it's fun to see what words it thinks are positive or negative, and what words it thinks are and aren't similar. Last, collocations seems to just be a quick and easy way to find the bigrams in a text. I know it's trying to find collections that are greater than the sum of their parts but my results seem lacking.

You can see the Collab [here](WordNet/WordNetHw3.pdf)