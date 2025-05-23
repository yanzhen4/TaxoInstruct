import json
from tqdm import tqdm
import fire
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import os

def preprocess_data(
    dataset: str = None,
    input_data_path: str = None,
    output_data_path: str = None,
    run_name: str = None
    ): 

    print(f'dataset: {dataset}')
    print(f'input_data_path: {input_data_path}')
    print(f'output_data_path: {output_data_path}')
    print(f'run_name: {run_name}')

    def phrase_parent(response):
        response_text = response.split("### Response:")[-1].strip()
        response_text = response_text.replace('The parent class is ', '')
        response_text = response_text.replace('<|end_of_text|>', '')
        return response_text

    def phrase_input(text):
        pattern = r"find the parent class for (.+?)(?=\n|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return None

    # Create the directory if it doesn't exist
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    else:
        print(f'Folder already exists')

    parents = set()

    with open(f'{output_data_path}/find_children.txt', 'w') as fout1:
        with open(f'{output_data_path}/find_parent.txt', 'w') as fout2:
            with open(f'{input_data_path}') as fin:
                for line in tqdm(fin):
                    data = json.loads(line)
                    input = data['input']
                    output = data['response']
                    output_parent_class = phrase_parent(output)
                    input_entity = phrase_input(output)

                    #Case of insufficient input entities
                    if input_entity in parents:
                        continue

                    fout1.write(f'{input_entity}\n')
                    fout2.write(f'{input_entity}\t{output_parent_class}\n')
                    parents.add(output_parent_class)

if __name__ == "__main__":
    fire.Fire(preprocess_data)
