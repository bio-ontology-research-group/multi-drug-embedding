
import pdb
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import random
import pandas as pd
import json

np.random.seed(33)

data_dir = 'data/'
mapping = pd.read_csv(data_dir+'mapping_WalkingRDFOWL.txt', header = None, sep = '\t')
mapping = mapping.values
mapping_dict = dict(zip(mapping[:,1],mapping[:,0]))


graph = [line.strip().split() for line in open(data_dir+"edgelist_WalkingRDFOWL.txt")]
sub_links = [(item[0],item[1],item[2]) for item in graph if 'has_indication' in mapping_dict[int(item[2])]]




graph = [(item[0],item[1],item[2]) for item in graph]
graphorig = set(graph) 

print('All graph edges before dropping: {}'.format(len(graph)))
print('sub graph edges: {}'.format(len(sub_links)))
data = np.array(sub_links)

test = [(item[0],item[1],item[2]) for item in data]
rdfgraph = set(graph) - set(test)


newrdf = list(rdfgraph)
newrdf = np.array(newrdf)

test = np.array(test)

edgelist = data_dir+ 'edgelist_WalkingRDFOWL_has_indication_free.txt'

np.savetxt(edgelist, newrdf, fmt = '%s')
pdb.set_trace()
