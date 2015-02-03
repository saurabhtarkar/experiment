#!python

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('deerwester.dict')#directly loading the initally computed dictionay
corpus = corpora.MmCorpus('deerwester.mm')  #this is the vector corpus

#to check you can go for

#print(corpus)
#MmCorpus(17 documents, 195 features, 259 non-zero entries)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

#to get the feel for matrix go for
# for doc in corpus_tfidf:
#  ...:  print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=195)
corpus_lsi = lsi[corpus_tfidf]

#now again test 
# lsi.print_topics(2)

lsi.save('model.lsi')




