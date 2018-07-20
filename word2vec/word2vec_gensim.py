import numpy as np 
import gensim 
import pdb
#from word2veckeras.word2veckeras import Word2VecKeras
Data_ = "../data/"
walks_file = Data_ + 'corpus_WalkingRDFOWL_has_indication_free.txt'
emeddings_file = Data_ + 'embeddings_WalkingRDFOWL_has_indication_free.txt'

print('reading the walks file....')
sents=gensim.models.word2vec.LineSentence(walks_file)
#pdb.set_trace()
print('training word2vec gensim model...')
model = gensim.models.word2vec.Word2Vec(sents,size=128, window=10, min_count=1, sg =1, workers=24)

model.wv.save_word2vec_format(emeddings_file)
