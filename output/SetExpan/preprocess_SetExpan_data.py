import json
from tqdm import tqdm
import fire
import re
import numpy as np

def load_parents(parent_data_path: str = None):
    parents = []
    with open(f'{parent_data_path}') as fin:
        for line in fin:
            parent = line.strip('\n')
            parents.append(parent)
    return parents

def preprocess_SetExpan(
    parent_data_path: str = None,
    setexpan_input_data_path: str = None,
    output_data_path: str = None,
    ):
    
    print(f'parent_data_path: {parent_data_path}')
    print(f'setexpan_input_data_path: {setexpan_input_data_path}')
    print(f'output_data_path: {output_data_path}')

    parents = load_parents(f'{parent_data_path}/parents.txt')
    
    idx = 0
    with open(f'{output_data_path}', 'w') as fout:
        with open(f'{setexpan_input_data_path}') as fin:
            for line in fin:
                data = json.loads(line)
                output_dict = data
                output_dict['parent'] = parents[idx]
                output_dict['task'] = 'Set-Expan'
                idx += 1
                json.dump(output_dict, fout)
                fout.write('\n')


if __name__ == "__main__":
    fire.Fire(preprocess_SetExpan)
