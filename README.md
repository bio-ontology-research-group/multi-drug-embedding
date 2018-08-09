# Drug repurposing through joint learning on knowledge graphs and literature

Here, we developed a novel method that combines
  information in literature and structured databases, and applies
  feature learning to generate vector space embeddings. We apply our
  method to the identification of drug targets and indications for
  known drugs based on heterogeneous information about drugs, target
  proteins, and diseases.  We demonstrate that our method is able to
  combine complementary information from both structured databases and
  from literature.

Below are the steps for the drugs repurposing pipleine

## Requirements
* python 2.7.6
* numpy 
* [keras](https://keras.io/)
* [boost libraries](https://www.boost.org/) for running multithreaded implementations of randomwalk.


## Running

1. Build the graph as described in [link](https://academic.oup.com/bioinformatics/article/3760100/Neuro-symbolic-representation-learning-on)

2. The output graph is in the data folder in this repository

3. Before generating the corpus, remove the `has-target` edges for (Drug target interactions) prediction, and `has-indication` edges for Drug indications prediction.

~~~~
python remove_relation_links.py
~~~~
4. Generate the knowledge graph corpus from the edgelist after removing edges, run

~~~~
./deepwalk ../data/edgelist_WalkingRDFOWL_has_indication_free.txt ../data/corpus_WalkingRDFOWL_has_indication_free.txt
~~~~

5. Run word2vec on the generated corpus
~~~~
python word2vec_gensim.py
~~~~

6. Normalize the knowledge graph entities with the PubMed abstracts corpus by running
~~~~
python normalize_text.py
~~~~
7. Use the the generated corpus from step 5 with Word2Vec to create independent Pubmed abstracts embeddings. 

8. Combine the generated corpus from step 5 with the knowledge graph corpus similar to the following and run Word2Vec on the combined corpus.

~~~~
cat ../data/corpus_WalkingRDFOWL_has_indication_free.txt ../data/medline_abstracts_mapped_drugsrepo.txt > ../data/combined_corpus.txt
~~~~
9. Run word2Vec on the combined corpus.
10. Run `Ind_ann_graph_common.py` and other scripts to train the Artificaial Neural Networks with different embeddings from the knowledge graph and PubMed abstracts available in the data folder.

## Data
## Knowledge graph and literature
The PubMed abstarcts used in this project was downloaded from [Pubtator](ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/).
The knowledge graph edgelist is `edgelist_WalkingRDFOWL.txt` and the mapping to knowledge graph node is `mapping_WalkingRDFOWL.txt`

## Embeddings
`embeddings_WalkingRDFOWL_has_indication_free.txt` knowledge graph embeddings for predicting drug indications
`embeddings_WalkingRDFOWL_has_targets_free.txt` knowledge graph embeddings for predicting drugs targets 
`drugs_text_embeddings.txt`, `diseases_text_embeddings.txt` and `genes_text_embeddings.txt` are Medline abstracts embeddings.
`drugs_embeddings_combined_has_indication.txt`, `diseases_embeddings_combined_has_indication.txt` and `genes_embeddings_combined_has_indication.txt` are knowledge graph and Medline abstracts jointly trained.
## Evaluations and Mapping
  All generated embeddings and mapping data used to normalize Literature information to knowledge graph used in this project is available as python dictionary in the data folder. All drug indications `drugs2ind_doid.dict` and drug targets `drugs2tars_stitch.dict` evaluations are available as well.
The drug indications is from [SIDER](http://sideeffects.embl.de/) database. The drug target is from [STITCH](http://stitch.embl.de/) database. Chemicals alias from `STITCH` was used to convert drugs mentions in text to `STITCH` ID available in `chemical_map.dict`.

[Disease ontology](http://www.obofoundry.org/ontology/doid.html) was used to extract `MESH` to `DOID` mapping in `mesh2doid.dict` and `OMIM` to `DOID` in `omim2doid.dict`

## Predictions
We make drug indications predictions for approved drugs from SIDER available `predicted_indications_approved_processed.tsv` in the data folder.
The first column is the drug ID and drug name, indications disease ontology ID and name, and the prediction score. The full list of the tested drugs and the predicted ranks for indications and targets are included as `indications_ranked_graph.txt`, `indications_ranked_multimodalI.txt` and `indications_ranked_multimodalII.txt`, etc.
The first is the drug PubChem ID followed by the diseases and their ranks.

For the complete data including the embeddings, download from [here](http://bio2vec.net/data/drug-embeddings/)

## Citation 


