#!python

from gensim import corpora, models, similarities

stoplist = set('for a of the and to in'.split())

texts= [[word for word in doc.split() if word not in stoplist]
for doc in open('final_experimented.txt','r')]

dictionary = corpora.Dictionary(texts)	#here is the saved dictionary for the words and their tokensids.
dictionary.save('/tmp/deerwester.dict')

corpus = [dictionary.doc2bow(text) for text in texts]

corpus_tfidf = models.TfidfModel(corpus)
lsi =  lsi = models.LsiModel(corpus_tfidf,id2word=dictionary, numTopics=195)

#In [3]: dictionary = corpora.Dictionary.load('deerwester.dict')
#once you have the dictionary ready then you just simply load it into the corpus and start the computation;
#In [4]: corpus = corpora.MmCorpus('/tmp/deerwester.mm')

#this is the input from the otpt of the 
#In [5]: corpus = corpora.MmCorpus('deerwester.mm')
#In [6]: print(corpus)
#MmCorpus(17 documents, 195 features, 259 non-zero entries)

corpus_lsi = lsi[corpus_tfidf]
#tfidf.save('/tmp/tweets.tfidf_model')#here our enitire tifidf model is saved ceratainly in matrix form.

#so it looks somthing like this 
#In [40]: print(tfidf[corpus[5]])
#[(2, 0.0707361383849494), (4, 0.0937814144581064), (5, 9.160304115978176e-05), (18, 0.22178290773383086), (31, 0.07340211526032456), (75, 0.08082339602338522), (108, 0.18335878602604724), (109, 0.28369307039710334), (110, 0.17022204289857631), (111, 0.2203891850841044), (112, 0.2520411277406039), (113, 0.18335878602604724), (114, 0.16656696463466836), (115, 0.13319164384051912), (116, 0.2203891850841044), (117, 0.19483483338769447), (118, 0.38064939861508984), (119, 0.15431693093527513), (120, 0.17854759282871396), (121, 0.18873724242760487), (122, 0.28369307039710334), (123, 0.28369307039710334), (124, 0.28369307039710334), (125, 0.16318289073119496)]
