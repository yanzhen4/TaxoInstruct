import json
from tqdm import tqdm
import fire
import re
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel

def load_model(
    model_path: str = f'/shared/data2/yuz9/BERT_models/specter/',
    device = torch.device(0)
):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, output_hidden_states=True).to(device)
    model.eval()

    return model, tokenizer

def encode_text(
    text: str = None,
    model = None,
    tokenizer = None,
    device = torch.device(0)
):

    input_ids = torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)).unsqueeze(0).to(device)
    outputs = model(input_ids)
    hidden_states = outputs[2][-1][0]
    emb = torch.mean(hidden_states, dim=0).cpu()
    return emb.detach()

def cosine_similarity(
    embs1: list = None, 
    embs2: list = None
):
    embs1 = np.array(embs1) / np.linalg.norm(embs1)
    embs2 = np.array(embs2) / np.linalg.norm(embs2)

    return np.dot(embs1, embs2)

def encode_entities(
    entities: list = None,
    model = None,
    tokenizer = None
):
    entities_emb = {}
    for entity in entities:
        emb = encode_text(entity, model, tokenizer)
        entities_emb[entity] = emb

    return entities_emb

def select_top_k_entities(
    target: str = None, 
    entities_emb: dict = None, 
    k: int = 1,
    model = None,
    tokenizer = None
):

    target_emb = encode_text(target, model, tokenizer)

    similarities = {}
    for entity in entities_emb:
        emb = entities_emb[entity]
        sim = cosine_similarity(target_emb, emb)
        similarities[entity] = sim

    top_entities = [entity for entity, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]]

    return top_entities

def read_entities(
    entities_path: str = None,
    query_indices: list = None
):
    idx2entities = {}

    for i in range(len(query_indices)):
        query_idx = query_indices[i]
        entities = []
        with open(f'{entities_path}/{i+1}.txt') as fin:
            for line in fin:
                entity = line.strip('\n')
                entities.append(entity)

        idx2entities[query_idx] = entities
    
    return idx2entities

def read_query_indices(
    query_indices_path: str = None, 
    dataset: str = 'MAG', 
):
    if query_indices_path:
        query_indices = []
        with open(f'{query_indices_path}/supplement_query_indices.txt') as fin:
            for line in fin:
                query_idx = int(line.strip('\n'))
                query_indices.append(query_idx)
    else:
        dataset2query_count = {
            'MAG': 9,
            'APR': 15,
            'WIKI': 40
        }
        
        query_indices = list(range(dataset2query_count[dataset]))
        query_indices = [x + 1 for x in query_indices]

    return query_indices

def read_parents(
    parents_path: str = None
):
    parents = ['']
    with open(parents_path) as fin:
        for line in fin:
            parent = line.strip('\n')
            parents.append(parent)
    return parents

def concat_supplement(
    dataset: str = 'MAG',
    setexpan_output_folder: str = None,
):
    print("dataset:", dataset)
    print("setexpan_output_folder:", setexpan_output_folder)

    model, tokenizer = load_model()

    parents = read_parents(f'{setexpan_output_folder}/parents.txt')
    query_indices_firstrun = read_query_indices(None, dataset)
    idx2entities_firstrun = read_entities(f'{setexpan_output_folder}/first_run', query_indices_firstrun)

    query_indices_supplement = read_query_indices(f'{setexpan_output_folder}/supplement', None)
    idx2entities_supplement = read_entities(f'{setexpan_output_folder}/supplement', query_indices_supplement)

    for query_idx in tqdm(query_indices_firstrun):
        entities = idx2entities_firstrun[query_idx]
        if query_idx in idx2entities_supplement:
            entities += idx2entities_supplement[query_idx]
        entities = list(set(entities))
        entities_embs = encode_entities(entities, model, tokenizer)
        top_entities = select_top_k_entities(parents[query_idx], entities_embs, len(entities_embs), model, tokenizer)
        with open(f'{setexpan_output_folder}/{query_idx}.txt', 'w') as fout:
            for entity in top_entities:
                fout.write(entity + '\n')

if __name__ == "__main__":
    fire.Fire(concat_supplement)
