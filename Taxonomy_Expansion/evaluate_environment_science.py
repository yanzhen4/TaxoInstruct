from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import json
import difflib
import networkx as nx

device = torch.device(3)

dataset = 'science'

bert_model = f'/shared/data2/yuz9/BERT_models/{model_name}/'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

'''
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model, output_hidden_states=True).to(device)
'''

folder = ''
data_folder = ''
file_name = f''

def specter_encode(text):
    input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
    outputs = model(input_ids)
    hidden_states = outputs[2][-1][0]
    emb = torch.mean(hidden_states, dim=0).cpu()
    return emb.detach()

def calculate_cosine_similarity(list1, list2):
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Calculate the dot product of the arrays
    dot_product = np.dot(array1, array2)

    # Calculate the norm (magnitude) of each array
    norm_array1 = np.linalg.norm(array1)
    norm_array2 = np.linalg.norm(array2)

    # Calculate cosine similarity
    similarity = dot_product / (norm_array1 * norm_array2)

    return similarity

def correct_spelling(word_dict, target_word):
    # Assuming you want to match against the keys of the dictionary
    keys_list = list(word_dict.keys())

    # Get the closest match to the target word from the keys of the dictionary
    closest_match = difflib.get_close_matches(target_word, keys_list, n=1, cutoff=0.6)

    if closest_match:
        return closest_match[0]  # Return the closest match
    else:
        return None  # Return None if no close match is found
        
def select_from_candidates(output, taxon2emb):

    output_emb = specter_encode(output)
    
    similarities = {}

    # Calculate cosine similarity for each entity
    for entity in taxon2emb:
        emb = taxon2emb[entity]
        sim = calculate_cosine_similarity(output_emb, emb)
        similarities[entity] = sim

    top_entity = max(similarities, key=lambda k: similarities[k])

    return top_entity

prefixes = [" the parent class is ", "the parentclass is ", " theparent class ", 
            "the parent classes is", "the parent classification is ", "the parent classification is ", 
            "the parent classes are ", "the parent category is ", "the parent category is ", 'The parent is ', 'Theparent class is ', 'The parent classes are ', 
            'It is the study of the relationship between ', 'The parentclass is ', 'The parent classes is ', 
            'The parent classes is ', 'It is in the ']

def remove_prefixes(s, prefixes = prefixes):
    
    s = s.lower()

    for prefix in prefixes:
        prefix = prefix.strip(' ').lower()
        if s.startswith(prefix):
            return s[len(prefix) + 1:]
            
    s = s.strip(' ')
    return s


taxon2emb = {}

taxon2emb = {}
with open(f"{data_folder}/{dataset}/{dataset}_{model_name}_emb.json") as fin:
    for line in tqdm(fin):
        data = json.loads(line)
        taxon_name = data['taxon_name']
        taxon_emb = data[f'{model_name}_emb']
        taxon2emb[taxon_name] = taxon_emb

prefixes = [" the parent class is ", "the parentclass is ", " theparent class ", 
            "the parent classes is", "the parent classification is ", "the parent classification is ", 
            "the parent classes are ", "the parent category is ", "the parent category is ", 'The parent is ', 'Theparent class is ', 'The parent classes are ', 
            'It is the study of the relationship between ', 'The parentclass is ', 'The parent classes is ', 
            'The parent classes is ', 'It is in the ', "is "]

correct_count = 0
count = 0

with open(f'{folder}/{file_name}') as fin:
    for line in tqdm(fin):
        data = json.loads(line)
        parent = data['parent']
        output = data['output']
        output = remove_prefixes(output)
        parent = parent.strip(' ')
        candidate_parents = data['candidate_parents']

        corrected_output = correct_spelling(taxon2emb, output)
        
        if corrected_output == None:
            corrected_output = select_from_candidates(output, taxon2emb)

        #print(parent, ",", corrected_output, ",", output, end = ' ')
        
        if corrected_output == parent:
            correct_count += 1
            
        count += 1

taxon2emb = {}
with open(f"{data_folder}/{dataset}/{dataset}_emb.json") as fin:
    for line in tqdm(fin):
        data = json.loads(line)
        taxon_name = data['taxon_name']
        taxon_emb = data['specter_emb']
        taxon2emb[taxon_name] = taxon_emb

taxon_names = set()
with open(f'{data_folder}/{dataset}/{dataset}_raw_en.taxo') as fin:
    for line in fin:
        data = line.strip('\n').split('\t')
        child_name = data[1]
        parent_name = data[2]
        taxon_names.add(child_name)
        taxon_names.add(parent_name)

class Taxon(object):
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name
        
    def __lt__(self, another_taxon):
        if self.level < another_taxon.level:
            return True
        else:
            return self.rank < another_taxon.rank

class Taxonomy(object):
    def __init__(self, name="", node_list=None, edge_list=None):
        self.name = name
        self.graph = nx.DiGraph()
        self.tx_id2taxon = {}
        self.root = None
        
    def __str__(self):
        return f"=== Taxonomy {self.name} ===\nNumber of nodes: {self.graph.number_of_nodes()}\nNumber of edges: {self.graph.number_of_edges()}"
    
    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        return self.graph.number_of_edges()
    
    def get_nodes(self):
        """
        return: a generator of nodes
        """
        return self.graph.nodes()
    
    def get_edges(self):
        """
        return: a generator of edges
        """
        return self.graph.edges()
    
    def get_root_node(self):
        """
        return: a taxon object
        """
        if not self.root:
            self.root = list(nx.topological_sort(self.graph))[0]
        return self.root
    
    def get_leaf_nodes(self):
        """
        return: a list of taxon objects
        """
        leaf_nodes = []
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0:
                leaf_nodes.append(node)
        return leaf_nodes
    
    def get_children(self, parent_node):
        """
        parent_node: a taxon object
        return: a list of taxon object representing the children taxons
        """
        assert parent_node in self.graph, "parent node not in taxonomy"
        return [edge[1] for edge in self.graph.out_edges(parent_node)]
    
    def get_parents(self, child_node):
        """
        child_node: a taxon object
        return: a list of taxon object representing the parent taxons
        """
        assert child_node in self.graph, "child node not in taxonomy"
        return [edge[0] for edge in self.graph.in_edges(child_node)]
    
    def get_siblings(self, child_node):
        """
        child_node: a taxon object
        return: a list of taxon object representing the sibling taxons
        """
        
        assert child_node in self.graph
        parents = self.get_parents(child_node)
        siblings = []
        for parent in parents:
            children = self.get_children(parent)
            for child in children:
                if child != child_node and child not in siblings:
                    siblings.append(child)
        return siblings
        
    def get_descendants(self, parent_node):
        """
        parent_node: a taxon object
        return: a list of taxon object representing the descendant taxons
        """
        assert parent_node in self.graph, "parent node not in taxonomy"
        return list(nx.descendants(self.graph, parent_node))
    
    def get_ancestors(self, child_node):
        """
        child_node: a taxon object
        return: a list of taxon object representing the ancestor taxons
        """
        assert child_node in self.graph, "child node not in taxonomy"
        return list(nx.ancestors(self.graph, child_node))

    def add_node(self, node):
        self.graph.add_node(node)
        
    def add_edge(self, start, end):
        """
        start: a taxon object
        end: a taxon object
        """
        self.graph.add_edge(start, end)

    #These functions are used to calculate Wu&P
    def get_depth(self, node):
        depth = 1
        while node != self.get_root_node():
            parents = self.get_parents(node)
            if not parents:  # In case the node is root or disconnected
                break
            node = parents[0]  # Assuming a single parent as it's a DAG
            depth += 1
        return depth

    def find_LCA(self, node1, node2):
            
        ancestors1 = set(self.get_ancestors(node1))
        ancestors1.add(node1)
        
        ancestors2 = set(self.get_ancestors(node2))
        ancestors2.add(node2)
        
        common_ancestors = ancestors1.intersection(ancestors2)
        # Choose the LCA with the greatest depth
        return max(common_ancestors, key=lambda node: self.get_depth(node), default=None)

    def wu_palmer_similarity(self, node1, node2):
        lca = self.find_LCA(node1, node2)
        if lca is None:
            return 0  # No similarity if no LCA

        depth_lca = self.get_depth(lca)
        depth_node1 = self.get_depth(node1)
        depth_node2 = self.get_depth(node2)

        return 2.0 * depth_lca / (depth_node1 + depth_node2)

data_folder = '/shared/data3/yanzhen4/Taxo_Set_expan/BoxTaxo/data'

taxon2emb = {}
with open(f"{data_folder}/{dataset}/{dataset}_emb.json") as fin:
    for line in tqdm(fin):
        data = json.loads(line)
        taxon_name = data['taxon_name']
        taxon_emb = data['specter_emb']
        taxon2emb[taxon_name] = taxon_emb

taxon_names = set()
with open(f'{data_folder}/{dataset}/{dataset}_raw_en.taxo') as fin:
    for line in fin:
        data = line.strip('\n').split('\t')
        child_name = data[1]
        parent_name = data[2]
        taxon_names.add(child_name)
        taxon_names.add(parent_name)

taxonomy = Taxonomy(name = dataset)
taxon_names = set()
tx_name2taxon = {}
tx_taxon2name = {}

with open(f'{data_folder}/{dataset}/{dataset}_raw_en.taxo') as fin:
    for line in tqdm(fin):
        data = line.strip('\n').split('\t')
        child_name = data[1]
        parent_name = data[2]
        
        if child_name not in tx_name2taxon: #Add new taxon
            child_taxon = Taxon(child_name)
            tx_name2taxon[child_name] = child_taxon
            tx_taxon2name[child_taxon] = child_name
            taxonomy.add_node(child_taxon)
        else:
            child_taxon = tx_name2taxon[child_name]

        if parent_name not in tx_name2taxon: #Add new taxon
            parent_taxon = Taxon(parent_name)
            tx_name2taxon[parent_name] = parent_taxon
            tx_taxon2name[parent_taxon] = parent_name
            taxonomy.add_node(parent_taxon)
        else:
            parent_taxon = tx_name2taxon[parent_name]

        taxonomy.add_edge(parent_taxon, child_taxon)

child_names_specter = set()
total_wu_p = 0

taxon2emb = {}
with open(f"{data_folder}/{dataset}/{dataset}_{model_name}_emb.json") as fin:
    for line in tqdm(fin):
        data = json.loads(line)
        taxon_name = data['taxon_name']
        taxon_emb = data[f'{model_name}_emb']
        taxon2emb[taxon_name] = taxon_emb
    
with open(f'{folder}/{file_name}') as fin:
    
    for line in tqdm(fin):
        data = json.loads(line)
        child_name = data['child']
        parent_name = data['parent']
        output = data['output']
        taxon_parent = tx_name2taxon[parent_name]

        output_name = remove_prefixes(output)
        
        top_parent = correct_spelling(taxon2emb, output_name)
        
        if top_parent in tx_name2taxon:
            taxon_top_parent = tx_name2taxon[top_parent]
        else:
            top_parent = select_from_candidates(output, taxon2emb)
            taxon_top_parent = tx_name2taxon[top_parent]
            
        total_wu_p += taxonomy.wu_palmer_similarity(taxon_parent, taxon_top_parent)

print("Dataset:", dataset)
print("Model:", model_name)
print("ACC:", correct_count / count, correct_count, count)
print("Wu&P: ", total_wu_p / count)
