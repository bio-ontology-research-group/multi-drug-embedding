import pdb
import json

data_dir = '../../Documents/drugs_data/'
abstract = []
terms_dict = {}		



with open(data_dir+'mesh2omim.dict') as f:
	mesh2omim = json.load(f)


with open(data_dir+'chemical_map.dict') as f:
	chemical_map = json.load(f)


count_abstracts = 0
count_dis = 0
dis = []
genes = []
human_genes = []
chem = []
count_gene = 0
count_chem = 0
mapped_chem = []
with open(data_dir+'medline_abstracts_mapped_drugsRepo.txt','w') as file1:
	with open('../../Documents/text_graph/bioconcepts2pubtator_offsets') as f:
		for line in f:
			line = line.strip()
			if line:
				if '|t|' in line or '|a|' in line:
					line = line.split('|')[-1]
					text = line.split()
					abstract.append(text)
					count_abstracts = count_abstracts+1
				else:
					toks = line.strip().split('\t')
					if toks[4] == 'Disease' and len(toks) > 5:
						terms_dict[toks[3]] = '<http://bio2vec.net/disease/'+toks[5]+'>'
						dis.append(toks[5])
					elif toks[4] == 'Chemical' and len(toks) > 5:
						mesh = toks[5].strip('MESH:')
						chem.append(mesh)
						if mesh in chemical_map:
							chem_id = chemical_map[mesh]
							mapped_chem.append(chem_id)
							terms_dict[toks[3]] = '<http://bio2vec.net/chem/'+chem_id+'>'
					elif toks[4] == 'Gene':
						terms_dict[toks[3]] = '<http://www.ncbi.nlm.nih.gov/gene/'+toks[5]+'>'
						genes.append(toks[5])
						if 'Tax' not in toks[5]:
							human_genes.append(toks[5])
			else:
				for item in abstract:
					newitem = []
					for it in item:
						if it in terms_dict:
							term = terms_dict[it]
							newitem.append(term)
							if 'http://bio2vec.net/disease/' in term:
								count_dis = count_dis+1
							if 'http://bio2vec.net/chem/' in term:
								count_chem = count_chem+1
							if 'http://www.ncbi.nlm.nih.gov/gene/' in term:
								count_gene = count_gene+1
						else:
							newitem.append(it)

					file1.write(' '.join(newitem)+'\n')
				abstract = []
				terms_dict = {}	

			# break	

print 'number of abstracts is: {}\n'.format(count_abstracts/2)
print 'number of disease mentions:{}\n'.format(count_dis)
print 'number of gene mentions:{}\n'.format(count_gene)
print 'number of chemicals mentions:{}\n'.format(count_chem)
print 'number of distinct diseases:{}\n'.format(len(set(dis)))
print 'number of distinct chemicals in mesh:{}\n'.format(len(set(chem)))
print 'number of distinct chemicals in stitch:{}\n'.format(len(set(mapped_chem)))
print 'number of distinct genes:{}\n'.format(len(set(genes)))
print 'number of distinct human genes:{}\n'.format(len(set(human_genes)))
