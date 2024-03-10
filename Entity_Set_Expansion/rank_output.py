from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import json

device = torch.device(2)

model = "allenai/specter"

folder = '/shared/data3/yanzhen4/TaxoInstruct/Entity_Set_Expansion'

device = torch.device(0)
bert_model = f'/shared/data2/yuz9/BERT_models/specter/'

tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)

'''
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model, output_hidden_states=True).to(device)
'''
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

seed_classes = ['heart disease', 'heart disease', 'vascualr diseases']

for index in range(3):
    
    entities = []
    with open(f'{folder}/Set_Expan_output_new.jsonl') as fin:
        line = fin.readlines()[index]
        data = json.loads(line)
        output = data['output']
        entities = entities + output

    class_emb = specter_encode(seed_classes[index])

    entity2sim = {}
    for entity in tqdm(entities):
        if len(entity) != 0:
            # Check if the entity is a substring of any already added entities
            entity_emb = specter_encode(entity)
            sim = calculate_cosine_similarity(entity_emb, class_emb)
            entity2sim[entity] = sim
            
    top_entities  = [k for k,v in sorted(entity2sim.items(), key=lambda item: item[1], reverse=True)][:50]

    def phrase(text):
        parts = text.split()

        try:
            # Identifying and converting the float and integer
            prob = float(parts[-2]) # Second last element is the float
            label = int(parts[-1])     # Last element is the integer
            entity = " ".join(parts[:-2])
        except:
            print(parts)
            label = int(parts[-1])     # Last element is the integer
            entity = " ".join(parts[:-1])
        
        return entity, label 

    with open(f'{folder}/label_{index}.txt', 'w') as fout:
        for top_entity in top_entities:
            #print(top_entity, '!')
            is_substring = any(top_entity in exist_entity for exist_entity in top_entities if top_entity != exist_entity)
            if not is_substring:
                fout.write(top_entity + '\t\n')