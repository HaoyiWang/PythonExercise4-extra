import nltk
from nltk.corpus import gutenberg
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim as gensimvis
import pyLDAvis

#pyLDAvis.enable_notebook()
#nltk.download('gutenberg')

alice_corpus = gutenberg.raw('carroll-alice.txt')

alice_tokens = nltk.word_tokenize(alice_corpus)
stopwords = set(nltk.corpus.stopwords.words('english'))
alice_tokens = [token for token in alice_tokens if token.lower() not in stopwords and token.isalpha()]

dictionary = corpora.Dictionary([alice_tokens])
corpus = [dictionary.doc2bow(alice_tokens)]

num_topics = 5
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=num_topics)

lda_vis = gensimvis.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary)

pyLDAvis.show(lda_vis)