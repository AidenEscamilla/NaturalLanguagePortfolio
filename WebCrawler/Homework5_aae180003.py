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



def filterPage(soup, outputFileName, filterString):
    with open(outputFileName, 'w') as f:
        for link in soup.find_all('a'):
            link_str = str(link.get('href'))
            print(link_str)
            if filterString in link_str: #or 'front' in link_str:
                if link_str.startswith('/url?q='):
                    link_str = link_str[7:]
                    print('MOD:', link_str)
                if '&' in link_str:
                    i = link_str.find('&')
                    link_str = link_str[:i]
                if link_str.startswith('http') and 'google' not in link_str:
                    f.write(link_str + '\n')

def filterPageAppend(soup, outputFileName, filterString, moreFilterString):
    with open(outputFileName, 'a') as f:
        for link in soup.find_all('a'):
            link_str = str(link.get('href'))
            print(link_str)
            if filterString in link_str and moreFilterString in link_str: #or 'front' in link_str:
                if link_str.startswith('/url?q='):
                    link_str = link_str[7:]
                    print('MOD:', link_str)
                if '&' in link_str:
                    i = link_str.find('&')
                    link_str = link_str[:i]
                if link_str.startswith('http') and 'google' not in link_str:
                    f.write(link_str + '\n')

def getSoupFromWebsite(urlString):
    Myheaders = {'User-Agent': 'AppleWebKit/537.36'}
    Myurl = urlString
    req = Request(url=Myurl, headers=Myheaders)
    html = urlopen(req).read().decode('utf8')
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style", "html.parser"]):
        script.extract()    # rip it out

    # extract text
    return soup


def generateLyricFiles(urlsFile):
    filesMade = []
   
    with open(urlsFile, 'r') as f:
        LyricUrls = f.read().splitlines()

    for url in LyricUrls:
        lyricSoup = getSoupFromWebsite(url)
        fileName = 'Lyrics: ' + lyricSoup.find('h1').getText() + '.txt'
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

def createTF(documentFile):
    tokens = []
    stop_words = set(stopwords.words('english'))
    tf_dict = {}
    with open(documentFile, 'r') as f:
        lines = f.read().splitlines()   
        for line in lines:      #for eevry line tokenize 
            tokens += word_tokenize(line.lower())
    
        tokens = [w for w in tokens if w.isalpha() and w not in stop_words] #get clean tokens
    
        token_set = set(tokens)
        tf_dict = {t:tokens.count(t) for t in token_set}   #create dict
        for token in tf_dict.keys():
            tf_dict[token] = tf_dict[token] / len(tokens)   #calc requency
    
    return tf_dict

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

    #create idf dict
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


def main():
    sia = SentimentIntensityAnalyzer()
    ssl._create_default_https_context = ssl._create_unverified_context

    '''
    Myurl = "https://genius.com/artists/Front-porch-step"
    

    soup = getSoupFromWebsite(Myurl)
    filterPage(soup, 'urls.txt', 'album')


    with open('urls.txt', 'r') as f:
        AlbumUrls = f.read().splitlines()

    for album in AlbumUrls:
        albumSoup = getSoupFromWebsite(album)
        print('\nNEW LINE\n')
        filterPageAppend(albumSoup, 'lyricPages.txt', 'Front', 'lyrics')
    '''



    #filesCreated = generateLyricFiles('lyricPages.txt')

    #with open('files.pickle', 'wb') as handle:
    #    pickle.dump(filesCreated, handle)

    with open('files.pickle', 'rb') as handle:
        filesCreated = pickle.load(handle)

    holdResults = createTF_IDF_TfxIdf(filesCreated)
    list_of_tfs = holdResults[0]
    idf_dictionary = holdResults[1]
    tf_idf_list = holdResults[-1]
    

    print('\nPrinting top 25 terms per document:\n\n')
    # find the highest tf-idf terms for each document
    for dic in tf_idf_list:
        docName = dic.get('Document')
        del dic['Document']
        doc_term_weights = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        print("\n",docName, ': ', doc_term_weights[:25])

    allWords = []
    for dictionary in list_of_tfs:
        allWords += dictionary.keys()
    fd = nltk.FreqDist(allWords)
    print('\n\nMy top 10 terms', fd.most_common(11))

    #build knowledge base
    sentences = []
    knowledge = {}
    scores = []
    for file in filesCreated:
        with open(file, 'r') as f:
            lines = f.read()
            sentences = sent_tokenize(lines)

            for sentence in sentences:
                #tokens = word_tokenize(sentence.lower())
                #tokens = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')] #get clean tokens
                #sentence = ' '.join(tokens)
                scores.append(sia.polarity_scores(sentence)["compound"])
        total = 0
        for score in scores:
            total += score
        
        knowledge[file] = total/len(sentences)
    
    counter = 0
    for pair in knowledge:
        print('\n', pair, ': SENTI SCORE ', knowledge.get(pair))

    
    with open('knowledgeBase_sentiment_Dict.pickle', 'wb') as handle:
        pickle.dump(knowledge, handle)

    

    quit()

if __name__ == '__main__':
    main()