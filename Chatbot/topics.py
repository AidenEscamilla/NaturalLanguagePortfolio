# gensim and nltk imports
import pickle
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from songs import Songs
import IPython
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
from pyLDAvis import gensim

'''
CODE FROM Karen Mazidi's NLP class, fitted to my database of documents
'''
# preprocess docs
def preprocess(databaseOfDocs, stopwords):
    """
    Tokenize, remove stopwords and non-alpha tokens.
    param: docs - a list of raw text documents
    return: a list of processed tokens
    """
    
    processed_docs = []
    songDocuments = databaseOfDocs.getAllWithLyrics()
    print('Preprocessing your songs...\n')
    for i, doc in enumerate(songDocuments):
        tokens = [t for t in word_tokenize(doc['lyrics'].lower()) if t not in stopwords
                 and t.isalpha()]
        processed_docs.append(tokens)
    return processed_docs


def main():
    SongDb = Songs()
    NUM_TOPICS = 4
    betterStopwords = [ 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is',
                       'that', 'on', 'were', 'was', 'you','ooh','me','my','we', 
                       'of', 'do', 'so', 'na', 'all', 'but', 'ah', 'i', 'she','her',
                       'your','t','la','m','wouldn','oh','ohh','if','re','ll','s','what',
                       'because','when', 'at','know','can','be','with','like','for','just',
                       'now','da','okay','no','this','could','tell','want','more','make',
                       'not','go','feel','yeah','get','they','got','he','she','down','up',
                       'wo','too','are', 'from', 'don', 'en', 'hey', 'gon', 'bit', 'que',
                        'jalebi', 'out','say','there','as','baby','did','where',
                        'have','one','how', 'neck', 'into', 'y', 'let', 'would', 'time']
    '''comment and uncomment betterStopWords to play with the topics and see what makes it through'''
    #betterStopwords = stopwords.words('english')
    #betterStopwords += ['like','yeah','ayy','fuck','ai','nigga','lil','ooh','oh','uh','got','la','de','el','te','na','en','lo','ey','que', 'wo', 'gon', 'ah', 'could', 'go','get','let','wo','take','say','make', 'know', 'want', 'da', 'jalebi', 'ohh']
    processedSongs = preprocess(SongDb, betterStopwords)
    with open('Songdocs.pickle', 'wb') as handle:
            pickle.dump(processedSongs, handle)
    with open('Songdocs.pickle', 'rb') as handle:
        processedSongs = pickle.load(handle)

    # the dictionary maps words to id numbers
    dictionary = corpora.Dictionary(processedSongs)
    #dictionary = []
    with open('dictionary.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle)
    with open('dictionary.pickle', 'rb') as handle:
        dictionary = pickle.load(handle)

    print('len of dictionary:', len(dictionary))
    #print('some items:', dictionary[0], dictionary[100])

    corpus = [dictionary.doc2bow(tokens) for tokens in processedSongs]

    # each doc in the corpus is now a bag of words
    # printing the first few 'words' in the bag of words confirms that word order is lost
    #print(corpus[0][:5])
    #print(dictionary[4], dictionary[2], dictionary[1])

    # build an LDA model
    lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
    print("LDA Model Results")
    #for i in range(NUM_TOPICS):
    #    print("\nTopic #%s:" % i, lda_model.print_topic(i, 6))


    for i in range(NUM_TOPICS):
        top_words = [t[0] for t in lda_model.show_topic(i, 6)]
        print("\nTopic", str(i), ':', top_words)

    Categories = []
    for i in range(NUM_TOPICS):
         top_words = [t[0] for t in lda_model.show_topic(i, 3)]
         Categories.append(','.join(top_words))
         print(Categories[i])
    
    # look at weights for top 10 words in topic 0
    #print(lda_model.show_topic(0, 10))


    print("LDA Model 1 Perplexity:", lda_model.log_perplexity(corpus))



    coherence1 = CoherenceModel(model=lda_model,
                           texts=processedSongs, dictionary=dictionary, coherence='c_v')
    print('Coherence score:', coherence1.get_coherence())

    #pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

    pyLDAvis.save_html(vis, 'LDA_Visualization4.html')


    # build an LSI model
    '''
    lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
    print("LSI Model Results")
    for i in range(NUM_TOPICS):
        print("\nTopic #%s:" % i, lsi_model.print_topic(i, 10))
    coherence3 = CoherenceModel(model=lsi_model,
                           texts=processedSongs, dictionary=dictionary, coherence='c_v')
    print('Coherence score:', coherence3.get_coherence())
    '''
    #vis = pyLDAvis.gensim.prepare(lsi_model, corpus, dictionary)

    #pyLDAvis.save_html(vis, 'LSI_Visualization4.html')


if __name__ == '__main__':
    main()