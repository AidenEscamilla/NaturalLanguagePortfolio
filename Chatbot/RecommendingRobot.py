import os
import sys  # to get the system parameter
import re
import sqlite3
import ssl
import random
from songs import Songs
from WebCrawlSpotify import lyricRecommendation
from WebCrawlSpotify import setup


#class Classifyer:
#    def __init__(self, connection):
#        self.connection = connection

    #def getCategory(category):


def randomClassifier(SongDatabase):
    song = input('Enter a song (e.x Radioactive by Imagine Dragons): ')
    songData = song.split(" by ")
    #store it
    #print(songData)
    songArtist = songData[0]
    songTitle = songData[1]
    #print it / testing
    print('you entered: ', songTitle, ':', songArtist)


    result = SongDatabase.getRandomRows()
    #result = connection.execute('SELECT * FROM Song ORDER BY RANDOM() LIMIT 5')
    print('\nHere are some random songs you might like!\n')
    for row in result:
        print(row['name'], ': ', row['artist'])

def categoryClassifier(SongDatabase):

    categories = ['rock', 'pop', 'disco', 'piano', 'scremo', 'ska', 'jazz']
    #uncomment this block if you're starting the data base from scratch!
    
    #Add arbitraty categories
    SongDatabase.connection.execute('Update Song SET Category = NULL')
    all = SongDatabase.getAllSongs()
    for row in all:
        SongDatabase.connection.execute('UPDATE Song SET Category = ? WHERE url = ?', [random.choice(categories), row['url']])
    SongDatabase.connection.commit()
    
    #Re comment above here
    print('Enter a category for song recommendations:\n\n')
    for cat in categories:
        print(cat)
    
    Category = input("Type Category: ").lower()
    result = SongDatabase.getCategoryRows(Category)
    if result == -1:
        return '\nI couldn\'t find that category\n'



    print('\nHere are some',Category, 'songs you might like!\n')
    numOutput = 10
    for row in random.choices(result, k=10):
        print(row['name'], ': ', row['artist'])
    
    return 'Finished category execution Properly'


def lyricClassifier(SongDatabase, firstTimeFlag):
    Song_wanting = input('Please enter a song you want recommendations for: ')
    Song_Artist_Wanting = input('Please enter the artist (type N/A for unknown or unwanted): ')
    lyricArgs = []
    
    if Song_Artist_Wanting.lower() == 'n/a':
        lyricArgs.append(Song_wanting)
    else:
        lyricArgs.append(Song_wanting)
        lyricArgs.append(Song_Artist_Wanting)

    returnMessage = lyricRecommendation(SongDatabase, lyricArgs, firstTimeFlag)
    print(returnMessage)




def main():
    SongDb = Songs()
    setup(SongDb)
    firstTime = True

    while True:
        print('Choose how you\'d like your recommendations!\n')
        classifierChoice = input('1. random\n2. by category\n3. by lyrics\n(type 1, 2, or 3): ')
        if classifierChoice == '1':
            randomClassifier(SongDb)
        elif classifierChoice == '2':
            print(categoryClassifier(SongDb))
        elif classifierChoice == '3':
            lyricClassifier(SongDb, firstTime)
            firstTime = False

        print('\nDo you want another recommendation?')
        keepPlaying = input('y/n: ').lower()
        if keepPlaying == 'n':
            print('Okay have a good day!')
            break
    quit()



if __name__ == '__main__':
    os.environ["SPOTIPY_CLIENT_ID"] = "PUBLIC"
    os.environ["SPOTIPY_CLIENT_SECRET"] = "SECRET"
    os.environ["SPOTIPY_REDIRECT_URI"] = "https://localhost:8888/callback"

    main()
