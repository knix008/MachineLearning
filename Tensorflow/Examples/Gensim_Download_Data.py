import gensim.downloader as api
from gensim.models import Word2Vec

dataset = api.load("text8")
model = Word2Vec(dataset)
model.save("data/text8-word2vec.bin")