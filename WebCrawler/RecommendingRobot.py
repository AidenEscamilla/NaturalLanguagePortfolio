import sys  # to get the system parameter
import re
import sqlite3
import ssl
import random
from songs import Songs

#class Classifyer:
#    def __init__(self, connection):
#        self.connection = connection

    #def getCategory(category):


def randomClassifier(SongDatabase):
    song = input('Enter a song (e.x Radioactive by Imagine Dragons): ')
    songData = song.split(" by ")
    #store it
    print(songData)
    songArtist = songData[0]
    songTitle = songData[1]
    #print it / testing
    print(songTitle, ':', songArtist)


    result = SongDatabase.getRandomRows()
    #result = connection.execute('SELECT * FROM Song ORDER BY RANDOM() LIMIT 5')
    print('\nHere are some songs you might like!\n')
    for row in result:
        print(row['name'], ': ', row['artist'])

def categoryClassifier(SongDatabase):

    categories = ['Rock', 'Pop', 'Disco', 'Piano', 'Scremo', 'Ska', 'Jazz']

    print('Enter a category for song recommendations:\n\n')
    for cat in categories:
        print(cat)
    
    Category = input("Type Category: ")
    result = SongDatabase.getCategoryRows(Category)

    print('\nHere are some songs you might like!\n')
    for row in result:
        print(row['name'], ': ', row['artist'])



def main():
    SongDb = Songs()

    #randomClassifier(SongDb)
    #categoryClassifier(SongDb)


    quit()



if __name__ == '__main__':
    main()