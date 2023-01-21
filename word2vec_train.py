from gensim.models import Word2Vec
import os

dataset_path = "dataset/amwiki.txt"

with open(dataset_path) as f:
    sentences = list(f.readlines())

tokenized_sentences = map(lambda sentence:sentence.split(),sentences)
model = Word2Vec(sentences=list(tokenized_sentences), vector_size=100, window=5, min_count=1, workers=4,max_vocab_size=10_000,epochs=20)

if not os.path.exists("logs/models"):
    os.makedirs("logs/models")
model.save("logs/models/word2vec.model")