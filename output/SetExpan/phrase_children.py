import json
import os
import re
import fire

def create_directory(
    path: str
):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f'{path} folder already exists')

def phrase_children(
    output: str
):
    pattern = r"The expanded entities are \{([^\}]*)\}"
    match = re.search(pattern, output)
    if match:
        entities = match.group(1).split(', ')
        return entities
    else:
        return []

def phrase_input_entities(
    input_string: str
):
    seeds_match = re.search(r'\{([^}]+)\}', input_string)
    if seeds_match:
        seeds = seeds_match.group(1)
        seeds_list = [seed.strip() for seed in seeds.split(',')]
        return seeds_list
    else:
        return ""

def process_input_data(
    input_data_path: str, 
    output_data_path: str
):
    idx = 0
    with open(f'{output_data_path}/queries.txt', 'w') as fout1:
        with open(f'{input_data_path}') as fin:
            for line in fin:
                data = json.loads(line)
                response = data['response']
                input = data['input']
                input_entities = phrase_input_entities(input)
                entities = phrase_children(response)
                with open(f'{output_data_path}/{idx + 1}.txt', 'w') as fout2:
                    for entity in entities:
                        fout2.write(entity + "\n")
                
                fout1.write('\t'.join(input_entities) + '\n')

                idx += 1

def load_original_data(
    original_data_path: str
):
    idx2input_dict = {}
    idx = 0
    with open(original_data_path) as fin:
        for line in fin:
            data = json.loads(line)
            if 'idx' in data:
                idx = data['idx']
            idx2input_dict[idx] = data
            idx += 1
    return idx2input_dict

def load_parent_data(
    parent_data_path: str
):
    idx2parent = {}
    idx = 0
    with open(f'{parent_data_path}/parents.txt') as fin:
        for line in fin:
            parent = line.strip('\n')
            idx2parent[idx] = parent
            idx += 1
    return idx2parent

def process_supplement_query(
    input_data_path: str, 
    insufficient_data_path: str, 
    original_data_path: str, 
    parent_data_path: str
):
    idx2input_dict = load_original_data(original_data_path)
    idx2parent = load_parent_data(parent_data_path)
    
    supplement_query_indices = []

    idx = 0
    with open(f'{insufficient_data_path}/supplement_query.jsonl', 'w') as fout1:
        with open(input_data_path) as fin:
            for line in fin:
                data = json.loads(line)
                response = data['response']
                entities = phrase_children(response)
                if len(entities) < 20:
                    supplement_query_indices.append(idx)
                    output_dict = idx2input_dict[idx]
                    output_dict['input'] = entities[:3] + output_dict['input']
                    output_dict['parent'] = idx2parent[idx]
                    output_dict['output'] = []
                    output_dict['idx'] = idx + 1
                    output_dict['task'] = 'Set-Expan'
                    json.dump(output_dict, fout1)
                    fout1.write('\n')
                idx += 1
    
    create_directory(f'{parent_data_path}/supplement')
    
    with open(f'{parent_data_path}/supplement/supplement_query_indices.txt', 'w') as fout2:
        for idx in supplement_query_indices:
            fout2.write(str(idx + 1) + '\n')

def phrase_data(
    original_data_path: str = None,
    input_data_path: str = None,
    output_data_path: str = None,
    insufficient_data_path: str = None,
    parent_data_path: str = None,
    run_name: str = None,
    supplement_query: bool = False
):
    print(f'original_data_path: {original_data_path}')
    print(f'input_data_path: {input_data_path}')
    print(f'output_data_path: {output_data_path}')
    print(f'insufficient_data_path: {insufficient_data_path}')
    print(f'parent_data_path: {parent_data_path}')
    print(f'run_name: {run_name}')
    print(f'supplement_query: {supplement_query}')

    create_directory(output_data_path)
    create_directory(insufficient_data_path)
    process_input_data(input_data_path, output_data_path)
    
    if supplement_query:
        process_supplement_query(input_data_path, insufficient_data_path, original_data_path, parent_data_path)

if __name__ == "__main__":
    fire.Fire(phrase_data)
