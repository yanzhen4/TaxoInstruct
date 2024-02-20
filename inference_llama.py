import os
import sys
import json
from tqdm import tqdm
import random

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from itertools import permutations

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def split_input_string(input_string):
    """
    Splits the input string into individual states.
    Assumes the input string is in the format "{'state1, state2, state3'}".
    """
    # Removing curly braces and splitting by comma
    entities = input_string.strip("{}").replace("'", "").split(", ")
    return entities

def split_output_string(output_string):
    """
    Splits the input string into individual states.
    Assumes the input string is in the format "{'state1, state2, state3'}".
    """
    # Removing curly braces and splitting by comma
    prefix = 'The expanded entities belonging to the category with the same granularity are '
    entities = output_string[len(prefix):].strip("{}").replace("'", "").split(", ")
    return entities

def remove_prefix_Taxo_expan(output_string):
    prefix_list = ["Theparent class is", "The parent class is", "The parent classification is", "The parents class is", "The entity is", "The entity belongs to the class of", "The entity is classified in the field of ", "The class is", "The parentclass is"]
    
    for prefix in prefix_list:
        if output_string.startswith(prefix):
            output_string = output_string[len(prefix):]
            break

    # delete the space at the beginning and the end
    output_string = output_string.strip()

    return output_string

def Set_Expansion_prompter(input_entities):

    input_string = "{" + ", ".join(input_entities) + "}"

    prompt = '<|system|>\nYou are a helpful assistant. The entities given by the user belong to a single class. Expand the entities given by user. \n'

    prompt += '<|user|>\n' + "The entity set is " + input_string + '\n'

    prompt += '<|assistant|>\n'

    return prompt

def Set_Expansion_parent_prompter(input_entities, parent):

    input_string = "{" + ", ".join(input_entities) + "}"

    prompt = '<|system|>\nYou are a helpful assistant. Given a category and an entity set belonging to this category, output other entities belonging to this category and sharing the same granularity as the seeds. \n'

    prompt += '<|user|>\n' + 'Find other entities belonging to category ' + parent + ' and sharing the same granularity as the seeds' + input_string + "\n"


    prompt += '<|assistant|>\n'

    return prompt

# Predict parent class from parent class and its siblings
def Taxonomy_Expansion_prompter(child, candidate_parents):

    candidate_parent_string = "{" + ", ".join(candidate_parents) + "}"

    prompt = '<|system|>\nYou are a helpful assistant. Given a set of candidate parent classes: ' + candidate_parent_string + ' output the most likely parent class for the entity given by user. \n'

    prompt += '<|user|>\n Find the parent for ' + child + '\n'

    prompt += '<|assistant|>\n'

    return prompt

def Find_Parent_prompter(child):

    input_string = "{" + ", ".join(child) + "}"

    prompt = '<|system|>\nYou are a helpful assistant. Given a list of entities, output the most likely parent class for the entity given by user. \n'

    prompt += '<|user|>\n Find the parent for ' + input_string + '\n'

    prompt += '<|assistant|>\n'

    print(prompt)

    return prompt

def check_file_path(path):
    return os.path.exists(path)

def main(
    load_8bit: bool = False,
    base_model: str = "",
    taxo_full_entity: str = "",
    use_chat_prompt: bool = True, # whether to use the prompt for multi-turn conversation
    lora_weights: str = "Set_Expan_output/checkpoint-58",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca., 
    data_path: str = "",
    output_name: str = "",
    load_weights: bool = True,
    full_parent : bool = False,
    complete_output: bool = False,
    set_expan_parent: bool = False
):
   
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"taxo_full_entity: {taxo_full_entity}\n"
            f"lora_weights: {lora_weights}\n"
            f"output_name: {output_name}\n"
            f"data_path: {data_path}\n"
            f"full_parent: {full_parent}\n"
            f'load_weights: {load_weights}\n'
            f"complete_output: {complete_output}\n"
            f"set_expan_parent: {set_expan_parent}\n"
        )

    lora_weights_path = '/shared/data3/yanzhen4/Taxo_Set_expan/On-Demand-IE-Environ/training/' + lora_weights
    inference_data_path = '/shared/data3/yanzhen4/Taxo_Set_expan/On-Demand-IE-Environ/training/' + data_path
    if check_file_path(lora_weights_path) == False: 
            raise FileNotFoundError(f"File not found: {lora_weights_path}")
    if check_file_path(inference_data_path) == False: 
            raise FileNotFoundError(f"File not found: {inference_data_path}")

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if load_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        input=None,
        task='Set-Expan',
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        # do_sample=True,
        max_new_tokens=2048,
        **kwargs,
    ):
        
        if task == "Set-Expan":
            if set_expan_parent == False:
                prompt = Set_Expansion_prompter(input[0])
            else:
                prompt = Set_Expansion_parent_prompter(input[0], input[1])
        elif task == "Taxo-Expan":
            prompt = Taxonomy_Expansion_prompter(input[0],input[1])
        else:
            prompt = Find_Parent_prompter(input)


        #print(prompt)

        inputs= tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        output = prompter.get_response(output, use_chat_prompt=use_chat_prompt)

        return output
    
    folder = '/shared/data3/yanzhen4/Taxo_Set_expan/On-Demand-IE-Environ/training'
    with open(f'{folder}/{output_name}', 'w') as fout:
        with open(f'{folder}/{data_path}') as fin:
            Lines = fin.readlines()
            for line in tqdm(Lines):
                data = json.loads(line)
                
                task = data['task']

                if task == "Set-Expan":

                    input = data['input']

                    if 'parent' in data:
                        parent = data['parent']
                    else:
                        parent = None

                    if complete_output == True:
                        
                        print("Completing output")

                        unique_entities = set()
                        # for entity_order in permutations(input):
                            
                        #     output = evaluate(input = (entity_order, parent), task = task)
                        #     output = split_output_string(output)
                            
                        #     unique_entities.update(output)

                        #     if len(unique_entities) >= 200:
                        #         break
                        entities = input
                        iteration = 0
                        while(len(unique_entities) < 400 and iteration < 80):
                            random.shuffle(entities)
                            output = evaluate(input = (entities, parent), task = task)
                            output = split_output_string(output)
                            unique_entities.update(output)
                            iteration += 1
                            print(len(unique_entities), "unique entities expanded")

                        output = list(unique_entities)
                    else:
                        output = evaluate((input, parent))
                        output = split_output_string(output)
                    
                    output_dict = {'input': input, 'parent': parent, 'output': output}
                elif task == "Taxo-Expan":

                    child = data['child']
                    if 'parent' in data:
                        parent = data['parent']
                    else:
                        parent = None

                    candidate_parents = data['candidate_parents']

                    output = evaluate(input = (child, candidate_parents), task = task)

                    output_dict = {'child': child, 'parent': parent, 'output': output, 'candidate_parents': candidate_parents}
                
                elif task == "Find-Parent":
                    child = data['child']
                    output = evaluate(input = child, task = task)
                    output_dict = {'child': child, 'parent': output}
                else:
                    parent = data['parent']
                    output = evaluate(input = parent, task = task)
                    children = split_output_string(output)
                    output_dict = {'parent': parent, 'children': children}

                json.dump(output_dict, fout)
                fout.write('\n')

if __name__ == "__main__":
    fire.Fire(main)
