from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import json

device = torch.device(2)

model = "allenai/specter"

folder = ''

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model, output_hidden_states=True).to(device)

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

stopwords = ['are', ' is ', '{', 'and', 'by', 'with', ':', ' to ', ',', ' a ', 'type', ' the ', 
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', 'or']

def phrase_output_string(s, stopwords):

    if len(s) > 30 or len(s) < 5:
        return ''
        
    last_pos = -1

    for stopword in stopwords:
        pos = s.rfind(stopword)
        if pos > last_pos:
            last_pos = pos
            last_stopword_length = len(stopword)
            
    if last_pos != -1:
        return s[last_pos + last_stopword_length:].strip()
    else:
        return s

index = 0

seed_class = 'cardiovascular disease'

class_emb = specter_encode(seed_class)

entities = []
with open(f'{folder}/Seed-Guided_Taxonomy_Construction/Set_Expan_Expand_entities_output.jsonl') as fin:
    lines = fin.readlines()
    for line in lines:
        data = json.loads(line)
        output = data['output']
        entities = entities + output

entity2sim = {}
for entity in tqdm(entities):
    entity = phrase_output_string(entity, stopwords)
    if len(entity) != 0:
        # Check if the entity is a substring of any already added entities
        is_substring = any(entity in added_entity for added_entity in entity2sim)
        if not is_substring:
            entity_emb = specter_encode(entity)
            sim = calculate_cosine_similarity(entity_emb, class_emb)
            entity2sim[entity] = sim
        
top_entities  = [k for k,v in sorted(entity2sim.items(), key=lambda item: item[1], reverse=True)][:50]

gt_parents = ['cardiovascular abnormalities', 'vascular disease', 'heart disease', 'cardiac disease']

with open(f'{folder}/Seed-Guided_Taxonomy_Construction/Taxo_expan_input.jsonl', 'w') as fout:
    for entity in top_entities:
        output_dict = {'child': entity, 'candidate_parents': gt_parents, 'task': 'Taxo-Expan'}
        json.dump(output_dict, fout)
        fout.write('\n')