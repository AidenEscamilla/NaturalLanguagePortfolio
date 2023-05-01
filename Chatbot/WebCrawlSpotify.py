import urllib
from urllib.request import urlopen
from urllib.request import Request
import ssl
from bs4 import BeautifulSoup
import re
import nltk #maybe un needed
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from songs import Songs # my class file
import math
from nltk.sentiment import SentimentIntensityAnalyzer
import sys  # to get the system parameter
import requests  #maybe un needed
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import unicodedata
import linecache
import os
import pickle
import sqlite3
from statistics import mode, median, mean

def processSongs(songList, SongDb):
    url = "https://genius.com/"
    db = SongDb

    for song in songList:
        #print(song)
        artist = song.get('artist')
        title = song.get('name')

        artist = re.sub('[/](?=[0-9])', '-', artist)
        artist = re.sub('[Ff]eat.*', '', artist)     #fixes formating from song titles
        artist = re.sub('\'|[?.!$+,/]|\(feat*\)|[()]', '', artist)
        artist = re.sub('[&]', 'and', artist)
        artist = re.sub('[:]', '-', artist)

        title = re.sub('[/](?=[0-9])', '-', title)
        title = re.sub('[Ff]eat.*', '', title)     #fixes formating from song titles
        title = re.sub('\'|[?.!,+$/]|\(feat*\)|[()]', '', title)     #fixes formating from song titles
        title = re.sub('[&]', 'and', title)
        title = re.sub('[:]', '-', title)

        artist = artist.split(' ')
        artist = '-'.join(artist).lower()
        artist = re.sub('-[*-]', '', artist)

        title = title.split(' ')
        title = '-'.join(title).lower()
        title = re.sub('-[*-]', '', title)

        tokens = (artist + ' ' + title).split()
        if len(tokens) != 0:
            urlString = url + tokens[0] +'-' + '-'.join(tokens[1:]).lower() + '-lyrics'
            urlString = re.sub('-[*-]', '', urlString)      #Fixes specific formatting for many spaces and a dash included in title 
            
            song['url'] = urlString
            db.insertSong(song)


def getSpotifyArtists(trackLimit):
    artist = []
    scope = "user-library-read"
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    results = sp.current_user_saved_tracks(limit=trackLimit)
    for item in results['items']:
        track = item['track']
        #print(track['artists'][0]['name'], " – ", track['name'], ' - ', track['album']['name'])
        #albums.append(track['album']['name'])
        artist.append(track['artists'][0]['name'])
        


    return [*set(artist)]


def getSpotifyAlbums():
    albumIds = []
    AlbumOffset = 0
    moreAlbums = True
    counter = 0
    scope = "user-library-read"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    while moreAlbums:
        results = sp.current_user_saved_albums(limit=10, offset=AlbumOffset)
        #print(results)
        if len(results['items']) < 10:
            moreAlbums = False

        for i, album in enumerate(results['items']):
            #print(i, ': ', album['album']['name'], '\n', album['album']['tracks']['href'], '\n') 
            albumIds.append(album['album']['id'])          
        #print('LINE: ', AlbumOffset/10)
        AlbumOffset += 10
        

    #print("ALBUM_IDS = ", albumIds)
    return albumIds


def getAlbumSongs(AlbumIDList, songDB):
    MyOffset = 0
    songs = []
    urlList = []
    songDict = {}
    temp = {}
    moreSongs = True
    scope = "user-library-read"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    for album in AlbumIDList:
        #print('PLAYLIST: ', playlist)
        
        while moreSongs:
            #print("ALBUM ID = ", album)
            results = sp.album_tracks(album_id=album, limit=10, offset=MyOffset)
            
            for item in results['items']:
                if item['is_local']:    #Skip local files
                    continue
                #print(item['name'], ': ', item['artists'][0]['name'])
                temp = {'name': item['name'], 'artist': item['artists'][0]['name']}
                songs.append(temp)

            if len(results['items']) < 10:
                MyOffset = 0
                moreSongs = False
            else:
                MyOffset += 10

        moreSongs = True


    #make a set of songs before processing
    setMaker = []
    for song in songs:
        if not song in setMaker:
            setMaker.append(song)

    songs = setMaker   #tested, it works
    #print("\nSONGS = \n", songs)
    processSongs(songs, songDB)
   
    #result = songDB.getAllSongs()
   
    return songs


def getSpotifyPlaylists():
    morePlaylists = True
    MyOffset = 0
    playlistsTracksUrls = []
    scope = "playlist-read-private"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    while morePlaylists:
        results = sp.current_user_playlists(limit=10, offset=MyOffset)
        
        if len(results['items']) < 10:
            morePlaylists = False

        for i, playlist in enumerate(results['items']):
                #print(i, ': ', playlist['name'], '\n', playlist['tracks']['href'], '\n') 
            playlistsTracksUrls.append(playlist['id'])          
        #print('LINE: ', MyOffset/10)
        MyOffset += 10

        
    #print("PLAYLISTS = ", playlistsTracksUrls)
    return playlistsTracksUrls



def getPlaylistSongs(trackUrlList, songDB):
    MyOffset = 0
    songs = []
    urlList = []
    songDict = {}
    temp = {}
    moreSongs = True

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    for playlist in trackUrlList:
        #print('PLAYLIST: ', playlist)
        
        while moreSongs:
            results = sp.playlist_tracks(playlist_id=playlist, fields='items(is_local,track(name,artists(name)))', limit=10, offset=MyOffset)
            
            for item in results['items']:
                if item['is_local']:    #Skip local files
                    continue
                #print(item['track']['name'], ': ', item['track']['artists'][0]['name'])
                temp = {'name': item['track']['name'], 'artist': item['track']['artists'][0]['name']}
                songs.append(temp)
        
            if len(results['items']) < 10:
                moreSongs = False
            else:
                MyOffset += 10

        moreSongs = True


    #make a set of songs before processing
    setMaker = []
    for song in songs:
        if not song in setMaker:
            setMaker.append(song)

    songs = setMaker   #tested, it works

    processSongs(songs, songDB)
   
    #result = connection.execute('SELECT * FROM Song')
    #for row in result.fetchall():
    #    print(row['name'], ' - ', row['artist'])

    return songs


def getSpotifySongs(SongDb):
    songs = []
    urlList = []
    songDict = {}
    moreSongs = True
    songOffset = 0
    scope = "user-library-read"
    counter = 0
    #CHANGE COUNTER FOR MORE SONGS IN LIBRARY
    
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    while moreSongs:
        #print('IN SONG LIBRARY')
        results = sp.current_user_saved_tracks(limit=25, offset=songOffset)
        for item in results['items']:
            track = item['track']
            temp = {'name': track['name'], 'artist': track['artists'][0]['name']}
            songs.append(temp)
            counter += 1
        if len(results['items']) < 10:# or counter >= 25:
            moreSongs = False
        else:
            #print(counter)
            #print('OffSet = ', songOffset, 'len(items): ', len(results['items']))
            #print(songs[9 + songOffset])
            songOffset += 25
        

    
    processSongs(songs, SongDb)    
    
    return songs



def filterPage(soup, outputFileName, filterArtist, filterSong):
    with open(outputFileName, 'w') as f:
        for link in soup.find_all('a'):
            link_str = str(link.get('href'))
            #print(link_str)
            tokens = filterSong.split('-')
            for token in tokens:
                if filterArtist in link_str and token in link_str and 'lyric' in link_str: #or 'front' in link_str:
                    print('FOUND: ' , link_str, '\n')
                    if link_str.startswith('/url?q='):
                        link_str = link_str[7:]
                        print('MOD:', link_str)
                    if '&' in link_str:
                        i = link_str.find('&')
                        link_str = link_str[:i]
                    if link_str.startswith('http') and 'google' not in link_str:
                        f.write(link_str + '\n')
                        continue

def filterPageAppend(soup, outputFileName, filterString, moreFilterString):
    with open(outputFileName, 'a') as f:
        for link in soup.find_all('a'):
            link_str = str(link.get('href'))
            print(link_str)
            if filterString in link_str and moreFilterString in link_str: #ex. and 'Boywithuke' in link_str:
                if link_str.startswith('/url?q='):
                    link_str = link_str[7:]
                    print('MOD:', link_str)
                if '&' in link_str:
                    i = link_str.find('&')
                    link_str = link_str[:i]
                if link_str.startswith('http') and 'google' not in link_str:
                    f.write(link_str + '\n')

def getSoupFromWebsite(urlString, OriginalTitle, OriginalArtist, songDB):

    Myheaders = {'User-Agent': 'AppleWebKit/537.36'}
    notFound = {'url': urlString, 'name': OriginalTitle, 'artist': OriginalArtist}

    artist = OriginalArtist.split(' ')
    artist = '-'.join(artist).lower()
    artist = re.sub('-[*-]', '', artist)
    artistJustInCase = artist

    title = OriginalTitle.split(' ')
    title = '-'.join(title).lower()
    title = re.sub('-[*-]', '', title)
    SongTitleJustInCase = title

    #tokens = (artist + ' ' + title).split()
    #urlString = url + tokens[0] +'-' + '-'.join(tokens[1:]).lower() + '-lyrics'
    #urlString = re.sub('-[*-]', '', urlString)      #Fixes specific formatting for many spaces and a dash included in title 



    Myurl = urlString
    if Myurl == None:
        return '-1' #Find out how a None object is getting here
    elif not Myurl.isascii():
        #print('in here: ', Myurl)
        temp = unicodedata.normalize('NFKD', Myurl).encode('Ascii', 'ignore')
        Myurl = temp.decode('utf-8')
        #print('result: ', Myurl)
    



    req = Request(url=Myurl, headers=Myheaders)
    #TRY CATCH 404 ERROR HERE
    try:
        html = urlopen(req).read().decode('utf-8')
        
    except urllib.error.HTTPError as errh:
 
        Myurl = 'https://genius.com/artists/' + artistJustInCase
        print('INVALID ULR, instead TRYING: ', Myurl)
        req = Request(url=Myurl, headers=Myheaders)
        try:
            html = urlopen(req).read().decode('utf-8')
            soup = BeautifulSoup(html, features="html.parser")
            for script in soup(["script", "style", "html.parser"]):
                script.extract()    # rip it out
            filterPage(soup, 'tryingArtist.txt', artistJustInCase, SongTitleJustInCase)
            
            trying = linecache.getline('tryingArtist.txt', 1)
            findingLyricsUrl = trying# + '-lyrics'

            if not findingLyricsUrl.startswith('http'):
                raise urllib.error.HTTPError(url=findingLyricsUrl, code=None, msg='empty', hdrs=Myheaders, fp=None)

            req = Request(url=findingLyricsUrl, headers=Myheaders)
            html = urlopen(req).read().decode('utf-8')
            soup = BeautifulSoup(html, features="html.parser")

            for script in soup(["script", "style", "html.parser"]):
                script.extract()    # rip it out

            return soup
        
        except urllib.error.HTTPError as e:
            notFound['error'] = 'urllib.error.HTTPError'
            #connection.execute('UPDATE Song_Not_Found SET name = ?, artist = ?, error = ? WHERE url= ?', (notFound['name'], notFound['artist'], notFound['error'], notFound['url']))
            songDB.insertToNotFound(notFound)
            return '-1'
        except UnicodeEncodeError as typo:
            notFound['error'] = 'UnicodeEncodeError'
            #connection.execute('UPDATE Song_Not_Found SET name = ?, artist = ?, error = ? WHERE url= ?', (notFound['name'], notFound['artist'], notFound['error'], notFound['url']))

            songDB.insertToNotFound(notFound)
            return '-1'
    
    #except requests.exceptions.ConnectionError as errc:
    #    print ("Error Connecting:",errc)
    #except requests.exceptions.Timeout as errt:
    #    print ("Timeout Error:",errt)
    #except requests.exceptions.RequestException as err:
    #    print ("OOps: Something Else",err)

    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style", "html.parser"]):
        script.extract()    # rip it out

    # extract text
    return soup


def generateLyricFiles(songDataBase, newSongsList):

    for song in newSongsList:
        #print('DictGet: ', songNotWorking.get(url))
        lyricSoup = getSoupFromWebsite(song.get('url'), song.get('name'), song.get('artist'), songDataBase)
        
        if lyricSoup == '-1':       #Skip 404's
            #print('skipped: ', url)
            continue
        
        if isinstance(lyricSoup, int):
            print(lyricSoup)
        containers = lyricSoup.findAll('div', {"data-lyrics-container": True})

        text = ''

        for content in containers:
            text += content.getText()

        #txt clean up
        text = re.sub('\[[^\]]*\]', '', text)               #Delete everything between [] including brackets like '[Verse 1]', '[Chrous]', ect.
        text = re.sub('(?<=[?!])(?=[A-Z])', '. ', text)      #fixes lines that end in ?
        text = re.sub('\'(?=[A-Z])', '. ', text)         #Fixes country ' thats used to start a word e.x: 'Cause
        text = re.sub('(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z])', '. ', text)         #space out text because the <br /> is thrown away leaving words touching and hard to tokenize. Buuut you can seperate by capital letters because every new line they capitalize
        text = text.replace('wanna', 'want to')         #Fix wanna to want to
        text = re.sub('[Cc]an\'t', 'can not', text)     #replace can't or Can't with can not because word tokenize stops reading past ' because it's not alpha
        text = text.replace('...', '. ')               #This line and the one below fix specific formatting found on the website. This fixes ellipses
        text = text.replace('Cause', 'Because')         #This fixes country grammar
        text = re.sub(' \(*x[0-9]\)*', '. ', text) #This fixes the '(x2)' text
        text = re.sub('x[0-9]', '. ', text)            #This fixes x2, x3, x4... ect when not in parenthesis
        #print(text)
        text = sent_tokenize(text)
        
        
        
        lyricsOutput = ''
        for sentence in text:
            lyricsOutput += sentence + " "

        temp = {'url': song.get('url'), 'lyrics': lyricsOutput}
        songDataBase.insertToLyrics(temp)
    


#Term frequency (TF) is how often a token appears in a document 
def createTF(songGiven, songDB):
    tokens = []
    stop_words = set(stopwords.words('english'))
    tf_dict = {}
    

    #cleanedLyrics = re.sub('!', '. ', songGiven['lyrics'])
    #cleanedLyrics = re.sub('?', '. ', cleanedLyrics)
    songGiven = songDB.getSpecificWithLyrics(songGiven['name'], songGiven['artist'])
    if songGiven == None:
        return '-1'
    cleanedLyrics = songGiven['lyrics'].split('.')

    for line in cleanedLyrics:      #for eevry line tokenize 
        tokens += word_tokenize(line.lower())
    
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words] #get clean tokens #This w.isalpha() messes up apostrophies need to fix that
    
    token_set = set(tokens)
    tf_dict = {t:tokens.count(t) for t in token_set}   #create dict
    for token in tf_dict.keys():
        tf_dict[token] = tf_dict[token] / len(tokens)   #calc requency
    
    return tf_dict
#tf * idf is similar to an inportance measure for a token based on your corpus (training data)
def create_tfidf(tf, idf):
    tf_idf = {}
    for t in tf.keys():
        if t == 'Document': #skip the title i added
            continue
        tf_idf[t] = tf[t] * idf[t] 
        
    return tf_idf


def createTF_IDF_TfxIdf(songDB, userSongList):

    userSongRows = []
    #come back and see if this is usless with a better query
    for song in userSongList:
        userSongRows.append(songDB.getTitleArtistSong(song['name'], song['artist']))
    num_docs = int(len(userSongRows))
    print(num_docs)
    #create tf dictionaries
    tf = []

    #result = con.execute('SELECT lyrics FROM Lyrics')
    #rows = result.fetchall()
    #for i, row in enumerate(rows):
    #    print(i, ':###### ', row['lyrics'])
    #maybe prob here
    
    for i, songRow in enumerate(userSongRows):
        if songRow == None:
            continue
        #print(i, 'SONG: ', songRow['name'], " - ", songRow['artist'])
        temp = createTF(songRow, songDB)
        if temp == '-1':
            continue
        temp['Document'] = songRow['url']
        tf.append(temp)
    
    #create vocab
    vocab = set()
    for dictionary in tf:
        vocab = vocab.union(set(dictionary.keys()))

    #create inverse document frequency (idf) dict. If a word appears in every document its common if its in only 1 or 2 its a rare term
    idf_dict = {}
    vocab_by_topic = []
    for d in tf:
        vocab_by_topic.append(d.keys())


    for term in vocab:
        temp = ['x' for voc in vocab_by_topic if term in voc]
        idf_dict[term] = math.log((1+num_docs) / (1+len(temp))) 


    #create tf-idf
    tf_idf_list = []
    for t in tf:
        temp = create_tfidf(t, idf_dict)
        temp['Document'] = t.get('Document')
        tf_idf_list.append(temp)
    

    return tf, idf_dict, tf_idf_list





#I just keep adding peoples spotify songs into this dictionary full of songs and the related score
def buildKnowledgeBase(connection, term_Importance_list):
    sia = SentimentIntensityAnalyzer()

    sentences = []
    #scores = []
    weightedScores = []

    #res = connection.execute('SELECT Count(*) FROM Song')
    #num_rows = int(res.fetchone()[0])
    #print('NUM_ROWS:', num_rows, '\n')

    res = connection.execute('SELECT * FROM Song INNER JOIN Lyrics ON Song.url = Lyrics.url')
    for i, song in enumerate(res.fetchall()):

        lines = song['lyrics']
        sentences = sent_tokenize(lines)

        for sentence in sentences:
            if len(sentences) == 0:
                continue

            tokens = word_tokenize(sentence.lower())
            tokens = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')] #get clean tokens

            tf_idf_weight_multiplyer = 1
            for token in tokens:        #(taken the same way as the function), multiply together term weights to get total sentence importance. Multiply sentence sentiment score by importance per sentence to get more accurate scores
                if token in term_Importance_list[i]:
                    tf_idf_weight_multiplyer *= term_Importance_list[i].get(token)       #Double check if this is getting the right token
                else:
                    tf_idf_weight_multiplyer *= 1/len(term_Importance_list[i])       #Add smoothing here and an if statment for tokens not found #this is bad smooothing

            weightedScores.append(sia.polarity_scores(sentence)["compound"] * tf_idf_weight_multiplyer)
            #scores.append(sia.polarity_scores(sentence)["compound"])
        
        #total = 0
        weightedTotal = 0
        #for score in scores:
            #total += score
        for Wscore in weightedScores:
            weightedTotal += Wscore

        if len(sentences) != 0:     #need this iff statment because some songs are found and files are made but they are instrumentals with no lyrics
            connection.execute('UPDATE Song SET sentiment_Score = ? WHERE url = ?', (weightedTotal/len(weightedScores), song['url']))
            connection.commit()
    #counter = 0
    #for pair in knowledge:
    #    print('\n', pair, ': SENTI SCORE ', knowledge.get(pair))

    

    #print('\n\nKNOWLEDGE BASE\n\n')
    #for pair in knowledge:
    #    print(pair, ' SCORE: ', knowledge.get(pair))


def lyricRecommendation(SongDatabase, SongWantingRecommendationFor, firstTimeFlag):
    
    dfRecSong = {}
    errMessage = 'Sorry, I could not find that song from your library'
    lyricsRow = []


    if len(SongWantingRecommendationFor) == 1:
        answer = SongDatabase.searchUserSong(SongWantingRecommendationFor[0])
        if answer == -1:
            return errMessage
        print(answer['name'], ': ', answer['artist'])
        lyricsRow = SongDatabase.getSpecificWithLyrics(answer['name'], answer['artist'])
        dfRecSong['text'] = lyricsRow['lyrics']
        dfRecSong['artist'] = lyricsRow['artist']
        dfRecSong['song'] = lyricsRow['name']
        dfRecSong['link'] = lyricsRow['url']
    elif len(SongWantingRecommendationFor) == 2:
        recommendationArtist = SongWantingRecommendationFor[1]
        answer = SongDatabase.searchUserSongAndArtist(SongWantingRecommendationFor[0], recommendationArtist)
        if answer == -1:
            return errMessage
        print(answer['name'], ': ', answer['artist'])
        lyricsRow = SongDatabase.getSpecificWithLyrics(answer['name'], answer['artist'])
        dfRecSong['text'] = lyricsRow['lyrics']
        dfRecSong['artist'] = lyricsRow['artist']
        dfRecSong['song'] = lyricsRow['name']
        dfRecSong['link'] = lyricsRow['url']

    if firstTimeFlag:

        conn = sqlite3.connect('Songs.db') 
            
        sql_query = pd.read_sql_query ('''
                                SELECT DISTINCT s.url AS link, name AS song, artist, lyrics AS text  FROM Song AS s INNER JOIN Lyrics AS l on s.url = l.url WHERE length(lyrics) > 0
                                ''', conn)

        dfUser = pd.DataFrame(sql_query, columns = ['artist', 'song', 'link', 'text',])
        print ('Your songs as a Panda df!:\n', dfUser)

        dfRecSong = pd.DataFrame(dfRecSong, columns = ['artist', 'song', 'link', 'text',], index=[0])
        df = pd.read_csv('spotify_millsongdata.csv')
        df = pd.concat([df, dfUser], ignore_index=True)
        df = pd.concat([dfRecSong, df], ignore_index=True)
        df.reset_index()
        #print(df.head)
        #maybe don't need this variable
        X = df.text
        #print(X[0])
                            #making max_df high gets rid of stopwords, can play wiht this variable and ngrams
        vectorizer = TfidfVectorizer(max_df=0.7, ngram_range=(2, 4))
        Xtfid = vectorizer.fit_transform(X)
        
        with open('millTfidVector.pickle', 'wb') as handle:
            pickle.dump(Xtfid, handle)
        with open('pdDataFrame.pickle', 'wb') as handle:
            pickle.dump(df, handle)
    '''
    # read the data
    df = pd.read_csv('spotify_millsongdata.csv')
    df = pd.concat([df, dfUser], ignore_index=True)
    X = df.text
    millSongInfo = df.song

    # use defaults
    vectorizer = TfidfVectorizer(max_df=0.4, ngram_range=(1, 3))

    # vectorize
    Xtfid = vectorizer.fit_transform(X) # fit the corpus data
    
    with open('millTfidVector.pickle', 'wb') as handle:
        pickle.dump(Xtfid, handle)
    
    with open('millTfidVector.pickle', 'rb') as handle:
        Xtfid = pickle.load(handle)
    
    print('Vector Shape: ', Xtfid.shape, '\n')
    #print('made your pickle sir')
    '''
    


    #UnComment this block below for your first time!!
    #edit: this block needs to run everytime unless I find out how to search the df for song name
    
                             
    
    #re-comment above after first run through now that your spotify has been vectorized with the dataset


    with open('millTfidVector.pickle', 'rb') as handle:
        Xtfid = pickle.load(handle)
    with open('pdDataFrame.pickle', 'rb') as handle:
        df = pickle.load(handle)

    #print('#####: ', df.loc[df['link'] == lyricsRow['url']])
    foundIndex = df.loc[df['link'] == lyricsRow['url']].index.values[0]
    recommendationsFound = cosine_similarity(Xtfid[foundIndex], Xtfid)
    #print(recommendationsFound)

    print('\nStdev: ', np.std(recommendationsFound))
    stdev = np.std(recommendationsFound)
    meanVar = mean(recommendationsFound.tolist()[0])
    print('mean: ', meanVar)
    medianVar = median(recommendationsFound.tolist()[0])
    print('Median: ', medianVar, '\n')
    #Uncomment below to see the cosine vector score if you're curious
    topTen = sorted(recommendationsFound[0], reverse=True)[2:12]
    #print(topTen)
    recommendations = []
    for i, vector in enumerate(recommendationsFound.tolist()[0]):
        if(vector in topTen):
            #print('vec: ', vector)
            recommendations.append([i, vector])
    #print(recommendations)

    indexList = []
    for pair in recommendations:
        indexList.append(pair[0])

    dataRec = []
    for indexRec in indexList[0:11]:
        #print(indexRec)
        dataRec.append(str(df.iloc[[indexRec]]['song'].values[0] + ': by ' + df.iloc[[indexRec]]['artist'].values[0]))
    
    print('\nTop ten songs based on lyrics:\n')
    for songRec in dataRec:
        print(songRec, '\n')

    return 'Finished lyric execution Properly'

def setup(databseSongs):
    AlbumIDList = getSpotifyAlbums()
    albumSongsList = getAlbumSongs(AlbumIDList, databseSongs)
    
    playlistIDList = getSpotifyPlaylists()
    playlistSongsList = getPlaylistSongs(playlistIDList, databseSongs)
    
    savedSongsList = getSpotifySongs(databseSongs)


    
    newUserSongs = albumSongsList + playlistSongsList + savedSongsList
    #generateLyricFiles(databseSongs, newUserSongs)

def main():
    SongDb = Songs()
    #setup(SongDb)
    
    #SongWantingRecommendationFor = sys.argv[1]  
    #lyricRecommendation(SongDb, SongWantingRecommendationFor)
        #artistList = getSpotifyArtists(50)
        #print(artistList, '\n', len(artistList))
    
        #albumList = getSpotifyAlbums(50)
        #print(albumList, '\n', len(albumList))
    '''
    AlbumIDList = getSpotifyAlbums()
    albumSongsList = getAlbumSongs(AlbumIDList, SongDb)
    
    playlistIDList = getSpotifyPlaylists()
    playlistSongsList = getPlaylistSongs(playlistIDList, SongDb)
    
    savedSongsList = getSpotifySongs(SongDb)


    
    newUserSongs = albumSongsList + playlistSongsList + savedSongsList
    
    with open('userSongs.pickle', 'wb') as handle:
        pickle.dump(newUserSongs, handle)
    
    with open('userSongs.pickle', 'rb') as handle:
        newUserSongs = pickle.load(handle)

    #pickled and opened because this is a long process and i wanted to test and play with the code-
    #-Without webcrawling and geneerating every run
    #CHECKPOINT

    #generateLyricFiles(SongDb, newUserSongs)
    
    #res = con.execute("SELECT * FROM Lyrics")
    #res = con.execute('SELECT COUNT(*) FROM Song')
    #for row in res.fetchall():
    #    print(row['url'], ': ', row['lyrics'], '\n\n')

    
    holdResults = createTF_IDF_TfxIdf(SongDb, newUserSongs)
    list_of_tfs = holdResults[0]
    idf_dictionary = holdResults[1]
    tf_idf_list = holdResults[-1]
    

    print('\nPrinting top 25 terms per document:\n\n')

    # find the highest tf-idf terms for each document
    #FOR PRINTING PURPOSES
    for dic in tf_idf_list:             #bottome two lines used for printing purposes but the document name is a string and doesn't compar with floats so i delete it before sorting
        docName = dic.get('Document')
        del dic['Document']
        doc_term_weights = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        #print("\n",docName, ': ', doc_term_weights[:25])
    
    allWords = []
    for dictionary in list_of_tfs:
        allWords += dictionary.keys()
    fd = nltk.FreqDist(allWords)
    print('\n\nMy top 10 terms', fd.most_common(11))
    #Document pops up but it is my palceholder element just to hold the titles of the songs it's not actually the top word

    '''
    
    quit()
    '''
    #build knowledge base
    buildKnowledgeBase(con, tf_idf_list)
    
    result = con.execute('SELECT * FROM Song')
    for row in result.fetchall():
        print(row['name'], ': ', row['sentiment_Score'])
    
    quit()
    '''
if __name__ == '__main__':
    os.environ["SPOTIPY_CLIENT_ID"] = "9534b2ba3dd4408fbee3d0bf3cf4d912"
    os.environ["SPOTIPY_CLIENT_SECRET"] = "aa0326708f2f41fb90dcda386989de53"
    os.environ["SPOTIPY_REDIRECT_URI"] = "https://localhost:8888/callback"

    if len(sys.argv) < 2:
        print('Please enter a song you want recommendations for. (Type \'song_name\' \'song_artist\'(optional))\ne.x \'take me home country road\' \'John Denver\'')
        quit()
    else:
        print('All good, working now...') 
        main()










'''
    Myurl = "https://genius.com/artists/" + artistNameFromInput
    print('\n', Myurl)
    
    soup = getSoupFromWebsite(Myurl)
    filterPage(soup, 'urls.txt', 'album')

    
    with open('urls.txt', 'r') as f:
        AlbumUrls = f.read().splitlines()

    #Edit: nvm no need to be careful. this code clears the file before a new artist
    open('lyricPages.txt', "w").close()
    for album in AlbumUrls:
        albumSoup = getSoupFromWebsite(album)
        #print('\nNEW LINE\n')                         #CHANGE ARTIST NAME BELOW THE HASHTAG Edit: changed to sys arg input variable
        filterPageAppend(albumSoup, 'lyricPages.txt', artistNameFromInput, 'lyrics')
#'''
