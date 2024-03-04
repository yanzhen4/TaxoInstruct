from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import json

device = torch.device(2)

model = "allenai/specter"

folder = ''

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model, output_hidden_states=True).to(device)

# Set the model to evaluation mode
model.eval()

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

index = 0

seed_classes = ['cardiovascular abnormalities', 'vascualr diseases', 'heart disease']

class_emb = specter_encode(seed_classes[index])

entities = []
with open(f'{folder}/Set_Expan_output.jsonl') as fin:
    line = fin.readlines()[index]
    data = json.loads(line)
    output = data['output']
    entities = entities + output

entity2sim = {}
for entity in tqdm(entities):
    if len(entity) != 0:
        # Check if the entity is a substring of any already added entities
        is_substring = any(entity in added_entity for added_entity in entity2sim)
        if not is_substring:
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

entity2label = {}
with open(f'{folder}/label_all.txt') as fin:
    for line in tqdm(fin):
        entity, label = phrase(line)
        entity2label[entity] = label
        #print(entity, label)

with open(f'{folder}/label_{index}.txt', 'w') as fout:
    for top_entity in top_entities:
        if top_entity in entity2label:
            label = entity2label[top_entity]
            fout.write(top_entity + '\t' + str(label) + '\n')
        else:
            fout.write(top_entity + '\t\n')
