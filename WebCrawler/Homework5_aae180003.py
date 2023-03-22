import urllib
from urllib.request import urlopen
from urllib.request import Request
import ssl
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import math
from nltk.sentiment import SentimentIntensityAnalyzer
import sys  # to get the system parameter
import requests  
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import unicodedata
import linecache

 


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


def getSpotifyAlbums(trackLimit):
    albums = []
    scope = "user-library-read"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    results = sp.current_user_saved_tracks(limit=trackLimit)
    for item in results['items']:
        track = item['track']
        albums.append(track['album']['name'])
        


    return [*set(albums)]

def getSpotifyPlaylists():
    morePlaylists = True
    songOffset = 0

    scope = "playlist-read-private"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))


    while morePlaylists:
        results = sp.current_user_playlists(limit=10, offset=songOffset)
        
        if len(results['items']) < 10:
            morePlaylists = False
        else:
            for i, playlist in enumerate(results['items']):
                print(i, ': ', playlist['id'], '\n')            
            print('LINE: ', songOffset/10)
            songOffset += 10

        


    return [results]

def getSpotifySongs():
    songs = []
    urlList = []
    songDict = {}
    moreSongs = True
    songOffset = 0
    scope = "user-library-read"
    counter = 0

    #while moreSongs:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    while moreSongs:
        
        results = sp.current_user_saved_tracks(limit=10, offset=songOffset)
        for item in results['items']:
            track = item['track']
            temp = [track['name'], track['artists'][0]['name']]
            songs.append(temp)
            counter += 1
        if len(results['items']) < 10:
            moreSongs = False
        else:
            print('OffSet = ', songOffset, 'len(items): ', len(results['items']))
            print(songs[9 + songOffset])
            songOffset += 10
        

    Myurl = "https://genius.com/"
    

    for song in songs:
        song[1] = re.sub('[/](?=[0-9])', '-', song[1])
        song[1] = re.sub('[Ff]eat.*', '', song[1])     #fixes formating from song titles
        song[1] = re.sub('\'|[?.!$+,/]|\(feat*\)|[()]', '', song[1])
        song[1] = re.sub('[&]', 'and', song[1])
        song[1] = re.sub('[:]', '-', song[1])
        song[0] = re.sub('[/](?=[0-9])', '-', song[0])
        song[0] = re.sub('[Ff]eat.*', '', song[0])     #fixes formating from song titles
        song[0] = re.sub('\'|[?.!,+$/]|\(feat*\)|[()]', '', song[0])     #fixes formating from song titles
        song[0] = re.sub('[&]', 'and', song[0])
        song[0] = re.sub('[:]', '-', song[0])
        artist = song[1].split(' ')
        artist = '-'.join(artist).lower()
        artist = re.sub('-[*-]', '', artist)
        title = song[0].split(' ')
        title = '-'.join(title).lower()
        title = re.sub('-[*-]', '', title)
        tokens = (song[1] + ' ' + song[0]).split()
        urlString = Myurl + tokens[0] +'-' + '-'.join(tokens[1:]).lower() + '-lyrics'
        urlString = re.sub('-[*-]', '', urlString)      #Fixes specific formatting for many spaces and a dash included in title 
        urlList.append(urlString)
        songDict[urlString] = [artist, title]
        
        
    return urlList, songDict



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

def getSoupFromWebsite(urlString, songNotWorking):

    Myheaders = {'User-Agent': 'AppleWebKit/537.36'}
    artistJustInCase = songNotWorking[0]
    SongTitleJustInCase = songNotWorking[1]

    Myurl = urlString
    if not Myurl.isascii():
        print('in here: ', Myurl)
        temp = unicodedata.normalize('NFKD', Myurl).encode('Ascii', 'ignore')
        Myurl = temp.decode('utf-8')
        print('result: ', Myurl)



    req = Request(url=Myurl, headers=Myheaders)
    #TRY CATCH 404 ERROR HERE
    try:
        html = urlopen(req).read().decode('utf-8')
        
    except urllib.error.HTTPError as errh:
        #with open('NotWorkingLinks.txt', 'a') as f:
        #        f.write(Myurl + '\n')
        #return '-1'
        #CHECKPOINT
        Myurl = 'https://genius.com/' + artistJustInCase
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
            with open('NotWorkingLinks.txt', 'a') as f:
                f.write(urlString + '\n')
            return '-1'
        except UnicodeEncodeError as typo:
            with open('NotWorkingLinks.txt', 'a') as f:
                f.write(urlString + '\n')
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


def generateLyricFiles(urlsFile, songNotWorking):
    filesMade = []
   
    with open(urlsFile, 'r') as f:
        LyricUrls = f.read().splitlines()

    open('NotWorkingLinks.txt', "w").close()
    for i, url in enumerate(LyricUrls):
        print(url)
        print('DictGet: ', songNotWorking.get(url))
        lyricSoup = getSoupFromWebsite(url, songNotWorking.get(url))

        if lyricSoup == '-1':       #Skip 404's
            #print('skipped: ', url)
            continue

        fileName = 'Lyrics: ' + lyricSoup.find('h1').getText() + '.txt'
        fileName = fileName.replace('/', '_')
        fileName = 'Lyrics/' + fileName
        filesMade.append(fileName)
        containers = lyricSoup.findAll('div', {"data-lyrics-container": True})

        text = ''

        for content in containers:
            text += content.getText()

        #txt clean up
        text = re.sub('\[[^\]]*\]', '', text)               #Delete everything between [] including brackets like '[Verse 1]', '[Chrous]', ect.
        text = re.sub('(?<=[?])(?=[A-Z])', ' ', text)      #fixes lines that end in ?
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
        

        with open(fileName, 'w') as outputFile:
            for sentence in text:
                outputFile.write(sentence + " ")

    return filesMade

#Term frequency (TF) is how often a token appears in a document 
def createTF(documentFile):
    tokens = []
    stop_words = set(stopwords.words('english'))
    tf_dict = {}
    with open(documentFile, 'r') as f:
        lines = f.read().splitlines()   
        for line in lines:      #for eevry line tokenize 
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


def createTF_IDF_TfxIdf(filesIn):
    num_docs = len(filesIn)
    #create tf dictionaries
    tf = []
    for cleanFileName in filesIn:
        temp = createTF(cleanFileName)
        temp['Document'] = cleanFileName
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
def buildKnowledgeBase(filesCreated, term_Importance_list):
    sia = SentimentIntensityAnalyzer()

    sentences = []
    knowledge = {}
    #scores = []
    weightedScores = []

    #IF you have an error check here maybe you're reading before the knowledge base is created. Comment out and try again then uncomment out for future runs
    with open('knowledgeBase_sentiment_Dict.pickle', 'rb') as handle:
        knowledge = pickle.load(handle)

    for i, file in enumerate(filesCreated):

        with open(file, 'r') as f:

            lines = f.read()
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
            knowledge[file] = weightedTotal
    
    #counter = 0
    #for pair in knowledge:
    #    print('\n', pair, ': SENTI SCORE ', knowledge.get(pair))

    
    with open('knowledgeBase_sentiment_Dict.pickle', 'wb') as handle:
        pickle.dump(knowledge, handle)
    #print('\n\nKNOWLEDGE BASE\n\n')
    #for pair in knowledge:
    #    print(pair, ' SCORE: ', knowledge.get(pair))








def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    
        #num_songs = int(sys.argv[1])
    pickleFileName = sys.argv[1]
        #artistList = getSpotifyArtists(50)
        #print(artistList, '\n', len(artistList))
    
        #albumList = getSpotifyAlbums(50)
        #print(albumList, '\n', len(albumList))
    temp = getSpotifyPlaylists()
    '''
    urls_and_songs = getSpotifySongs()
    songUrlList = urls_and_songs[0]
    songMaybeNotWorking = urls_and_songs[1]
    with open('songs.pickle', 'wb') as handle:
        pickle.dump(songMaybeNotWorking, handle)
    with open('songs.pickle', 'rb') as handle:
        songMaybeNotWorking = pickle.load(handle)
    #songUrlList = []

    with open(pickleFileName, 'wb') as handle:
        pickle.dump(songUrlList, handle)
    
    with open(pickleFileName, 'rb') as handle:
        songUrlList = pickle.load(handle)
    
    #for song in songUrlList:
    #    print(song)
    

    
    with open('Lyrics/lyricPages.txt', 'w') as f:
        for url in songUrlList:
            f.write(url + '\n')
    

    #pickled and opened because this is a long process and i wanted to test and play with the code-
    #-Without webcrawling and geneerating every run
    #CHECKPOINT
    filesCreated = generateLyricFiles('Lyrics/lyricPages.txt', songMaybeNotWorking)
    
    with open('files.pickle', 'wb') as handle:
        pickle.dump(filesCreated, handle)
    
    with open('files.pickle', 'rb') as handle:
        filesCreated = pickle.load(handle)
    
    holdResults = createTF_IDF_TfxIdf(filesCreated)
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

    #build knowledge base
    buildKnowledgeBase(filesCreated, tf_idf_list)
    
    

    '''
    quit()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please enter  as a system arg  pickle file name. (Type \'FirstnameSongs.pickle\')')
        quit()
    else:
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


'''
