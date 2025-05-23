import json
from tqdm import tqdm
import fire
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def preprocess_data(
    dataset: str = None,
    SetExpan_output_data_path: str = None,
    TaxoExpan_input_data_path: str = None,
    ):

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

    def phrase_SetExpan_output(output):
        pattern = r"The expanded entities are \{([^\}]*)\}"
        match = re.search(pattern, output)

        if match:
            entities = match.group(1).split(', ')
            return entities
        else:
            return []
    
    def select_top_k_from_candidates(output, taxon2emb, k=10):
        output_emb = specter_encode(output)
        similarities = {}

        for entity, emb in taxon2emb.items():
            sim = calculate_cosine_similarity(output_emb, emb)
            similarities[entity] = sim

        top_k_entities = sorted(similarities, key=similarities.get, reverse=True)[:k]
        return top_k_entities

    idx = 0

    if dataset == 'dblp':
        parents = ["machine learning", "data mining", "natural language processing", "information retrieval", "wireless networks"]
    else:
        input_parents = ["vascular diseases", "heart disease", "cardiovascular abnormalities"]
        parents = ["vascular diseases", "heart disease", "cardiovascular abnormalities"]

    children = []

    with open(f'{SetExpan_output_data_path}') as fin:
        for line in fin:
            data = json.loads(line)
            output = phrase_SetExpan_output(data['response'])

            if idx == 0:
                for entity in output:
                    if entity not in parents:
                        parents.append(entity)
                idx += 1
            else:
                for entity in output:
                    if entity not in parents and entity not in children:
                        children.append(entity)

    taxon2emb = {}
    for parent in parents:
        parent_emb = specter_encode(parent)
        taxon2emb[parent] = parent_emb

    print("Parents: ", len(parents), parents)

    if dataset == 'dblp':
        parent_name = 'computer science'
    else:
        parent_name = 'cardiovascular disease'

    parent_emb = specter_encode(parent_name)
    child_similarities = []
    
    for child in children:
        child_emb = specter_encode(child)
        similarity = calculate_cosine_similarity(parent_emb, child_emb)
        child_similarities.append((child, similarity))
    
    sorted_children = [child for child, sim in sorted(child_similarities, key=lambda x: x[1], reverse=True)]

    all_top_k_parents = set()
    
    with open(f'{TaxoExpan_input_data_path}', 'w') as fout:
        for child in sorted_children:
            top_k_parents = select_top_k_from_candidates(child, taxon2emb, k=5)
            
            # Input classes already contain majority of cvd classes
            if dataset == 'cvd':
                top_k_parents = top_k_parents + input_parents

            all_top_k_parents.update(top_k_parents)
            output_dict = {"child": child, "parent": '', 'candidate_parents': top_k_parents, 'task': 'Taxo-Expan'}
            json.dump(output_dict, fout)
            fout.write('\n')

    # Add parents not in top_k_parents to the end of sorted_children list
    if len(sorted_children) < 50:
        with open(f'{TaxoExpan_input_data_path}', 'a') as fout:
            for parent in parents:
                if parent not in sorted_children:
                    top_k_parents = select_top_k_from_candidates(parent, taxon2emb, k=5)
                    output_dict = {"child": parent, "parent": '', 'candidate_parents': top_k_parents, 'task': 'Taxo-Expan'}
                    json.dump(output_dict, fout)
                    fout.write('\n')

if __name__ == "__main__":
    fire.Fire(preprocess_data)
