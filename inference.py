import wandb
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import fire
from tqdm import tqdm
import json
import random
import copy
import numpy as np

def set_seed(
    seed
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def inference_llama3(
    test_data_path: str = None,
    lora_weights: str = None,
    output_path: str = None,
    seed: int = 1
    ):

    print("test_data_path: ", test_data_path)
    print("lora_weights: ", lora_weights)
    print("output_path: ", output_path)
    print("seed: ", seed)

    set_seed(seed)

    if lora_weights:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = lora_weights,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
    else:
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     model_name = "unsloth/llama-3-8b-bnb-4bit",
        #     max_seq_length = 2048,
        #     dtype = None,
        #     load_in_4bit = True,
        #     device_map="auto"
        # )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen2-72B-Instruct-bnb-4bit",
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
            device_map="auto"
        )

    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    test_dataset = load_dataset("json", data_files=test_data_path)['train']

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}
    
    ### Response:
    """

    def formatting_prompts_func(
        example
    ):
        task   = example["task"]
        input  = example["input"]

        if task == 'Set-Expan':
            instruction = "Given a category and an entity set belonging to this category, output other entities belonging to this category and sharing the same granularity as the seeds."
        elif task == 'Taxo-Expan':
            instruction = "Given a set of candidate parent classes and an entity, output the most likely parent class for the entity given by user."
        elif task == 'Find-Parent':
            instruction = "Given a set of entities, output the most likely parent class for the entities given by the user"

        text = alpaca_prompt.format(instruction, input, '')

        return text

    pad_token_id = tokenizer.pad_token_id

    with open(output_path, "w") as f:
        for example in tqdm(test_dataset):
            if 'shuffle' in example:
                input_prompts = []
                for iteration in range(example['shuffle']):
                    example_copy = copy.deepcopy(example)
                    random.shuffle(example_copy['input'])
                    input_prompts.append(formatting_prompts_func(example_copy))
                
                inputs = tokenizer(input_prompts, return_tensors = "pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True, pad_token_id=pad_token_id)
                output_texts = tokenizer.batch_decode(outputs)

                for output_text in output_texts:
                    output_dict = example
                    output_dict['response'] = output_text

                    json.dump(output_dict, f)
                    f.write("\n")
            else:
                inputs = tokenizer([formatting_prompts_func(example)], return_tensors = "pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True, pad_token_id=pad_token_id)
                outputs_text = tokenizer.batch_decode(outputs)[0]
                output_dict = example
                output_dict["response"] = outputs_text

                json.dump(output_dict, f)
                f.write("\n")

if __name__ == "__main__":
    fire.Fire(inference_llama3)