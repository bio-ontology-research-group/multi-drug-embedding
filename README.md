# multimodal drugs embeddings


Below are the steps for the drugs multimodal repurposing pipleine


## The knowledge graph

1. Build the graph as described in  [link](https://academic.oup.com/bioinformatics/article/3760100/Neuro-symbolic-representation-learning-on)

2. The output graph is in the data folder in this repository

3. Before generating the corpus, remove the 'has-target' edges for (Drug target interactions) prediction, and 
(Drug-indications) edges for Drug indications prediction.

~~~~
Python remove_relation_links.py
~~~~
4. Generate the knowledge graph corpus from the edgelist after removing edges, run

~~~~
./deepwalk ../data/edgelist_WalkingRDFOWL_has_indication_free.txt ../data/corpus_WalkingRDFOWL_has_indication_free.txt
~~~~

5. Run word2vec on the generated corpus
~~~~
python word2vec_gensim.py
~~~~

## The PubMed abstracts

5. Normalize the knowledge graph entities with the PubMed abstracts corpus by running
~~~~
python normalize_text.py
~~~~

6. Run 'In_ann_graph_common.py' and other scripts to train the Artificaial Neural Networks with different embeddings from the knowledge graph and PubMed abstracts available in the data folder.

