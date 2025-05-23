import json
import fire
import re
import os
import numpy as np
from tqdm import tqdm

def phrase_parent(
    response: str = None
):
    response_text = response.split("### Response:")[-1].strip()
    response_text = response_text.replace('The parent class is ', '')
    response_text = response_text.replace('<|end_of_text|>', '')
    return response_text

def create_directory(
    path: str = None
):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f'{path} folder already exists')

def phrase_data(
    dataset: str = None,
    input_data_path: str = None,
    output_data_path: str = None,
    ):

    print(f'dataset: {dataset}')
    print(f'input_data_path: {input_data_path}')
    print(f'output_data_path: {output_data_path}')
    
    create_directory(output_data_path)

    with open(f'{input_data_path}') as fin:
        with open(f'{output_data_path}/parents.txt', 'w') as fout:
            for line in fin:
                data = json.loads(line)
                response = data['response']
                output_parent_class = phrase_parent(response)
                fout.write(f'{output_parent_class}\n')

if __name__ == "__main__":
    fire.Fire(phrase_data)
