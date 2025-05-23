import json
import fire
from transformers import BertTokenizer, BertModel
import torch
import networkx as nx
from tqdm import tqdm
import difflib
import re

device = torch.device(0)
bert_model = f'/shared/data2/yuz9/BERT_models/specter/'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

def specter_encode(text):
    input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
    outputs = model(input_ids)
    hidden_states = outputs[2][-1][0]
    emb = torch.mean(hidden_states, dim=0).cpu()
    return emb.detach()

def calculate_cosine_similarity(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)

    dot_product = np.dot(array1, array2)

    norm_array1 = np.linalg.norm(array1)
    norm_array2 = np.linalg.norm(array2)

    similarity = dot_product / (norm_array1 * norm_array2)

    return similarity

def correct_spelling(word_dict, target_word):
    keys_list = list(word_dict.keys())

    closest_match = difflib.get_close_matches(target_word, keys_list, n=1, cutoff=0.6)

    if closest_match:
        return closest_match[0]  # Return the closest match
    else:
        return None  # Return None if no close match is found

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

def evaluate(
    output_path: str = None,
    scores_path: str = None,
    raw_data_path: str = None,
    dataset: str = None,
    ):
    print("output_path: ", output_path)
    print("scores_path: ", scores_path)
    print("raw_data_path: ", raw_data_path)
    print("dataset: ", dataset)

    def phrase_response(response):
        response_text = response.split("### Response:")[-1].strip()
        response_text = response_text.replace('<|end_of_text|>', '')
        response_text = response_text.replace('The parent class is ', '')
        response_text = response_text.replace('</s>', '')
        response_text = response_text.replace('<eos>', '')

        return response_text
        
        #match = re.search(r'\bis\s+(.*)', response)
        #return match.group(1).strip() if match else output

    def phrase_output(output):
        output_text = output.replace("The parent class is ", '')
        return output_text

    taxon_names = set()
    with open(f'{raw_data_path}/{dataset}/{dataset}_raw_en.taxo') as fin:
        for line in fin:
            data = line.strip('\n').split('\t')
            child_name = data[1]
            parent_name = data[2]
            taxon_names.add(child_name)
            taxon_names.add(parent_name)

    taxon2emb = {}
    with open(f"{raw_data_path}/{dataset}/{dataset}_specter_emb.jsonl") as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            taxon_name = data['taxon_name']
            taxon_emb = data[f'specter_emb']
            taxon2emb[taxon_name] = taxon_emb

    taxonomy = Taxonomy(name = dataset)
    taxon_names = set()
    tx_name2taxon = {}
    tx_taxon2name = {}
    with open(f'{raw_data_path}/{dataset}/{dataset}_raw_en.taxo') as fin:
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

    total_wu_p = 0
    correct = 0
    total = 0
    with open(output_path) as fin:
        for line in fin:
            data = json.loads(line)
            response = phrase_response(data['response'])
            output = phrase_output(data['output'])
            taxon_parent = tx_name2taxon[output]
            top_parent = correct_spelling(taxon2emb, response)

            #Compute Wu&P
            if top_parent in tx_name2taxon:
                taxon_top_parent = tx_name2taxon[top_parent]
                total_wu_p += taxonomy.wu_palmer_similarity(taxon_parent, taxon_top_parent)
            else:
                total_wu_p += 0

            if response == output or top_parent == output:
                correct += 1
            total += 1

    #Two decimal places
    accuracy = round(correct / total * 100, 2)
    wu_p = round(total_wu_p / total * 100, 2)

    with open(scores_path, 'a') as fout:
       fout.write(f"{output_path}\t{accuracy}\t{wu_p}\n")

if __name__ == "__main__":
    fire.Fire(evaluate)