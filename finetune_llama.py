import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    PeftModel, 
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

def train(
    # model/data params
    base_model: str = "",  # the only required arguments
    taxo_full_entity: str = "",
    data_path: str = "data/train_data_Set_Expan.json",
    output_dir: str = "Set_expan_output_large",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,
    set_expan_parent: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"taxo_full_entity: {taxo_full_entity}\n"
            f"base_model_path: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"set_expan_parent: {set_expan_parent}\n"
        )
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def Set_Expansion_prompter(input_entities, output_entities):

        input_string = "{" + ", ".join(input_entities) + "}"
        output_string = "{" + ", ".join(output_entities) + "}"

        prompt = '<|system|>\nYou are a helpful assistant. The entities given by the user belong to a single class. Expand the entities given by user. \n'

        prompt += '<|user|>\n' + "The entity set is " + input_string + '\n'

        prompt += '<|assistant|>\n'

        user_prompt = prompt

        prompt += 'The expanded entities are ' + output_string

        return prompt, user_prompt

    def Set_Expansion_parent_prompter(input_entities, output_entities, parent):

        input_string = "{" + ", ".join(input_entities) + "}"
        output_string = "{" + ", ".join(output_entities) + "}"

        prompt = '<|system|>\nYou are a helpful assistant. Given a category and an entity set belonging to this category, output other entities belonging to this category and sharing the same granularity as the seeds. \n'

        prompt += '<|user|>\n' + 'Find other entities belonging to category ' + parent + ' and sharing the same granularity as the seeds' + input_string + "\n"

        prompt += '<|assistant|>\n'

        user_prompt = prompt

        prompt += 'The expanded entities belonging to the category ' + parent + ' and sharing the same granularity are ' + output_string

        print(prompt)
        
        return prompt, user_prompt

    # Predict parent class from parent class and its siblings
    def Taxonomy_Expansion_prompter(child, parent, candidate_parents):

        candidate_parent_string = "{" + ", ".join(candidate_parents) + "}"

        prompt = '<|system|>\nYou are a helpful assistant. Given a set of candidate parent classes: ' + candidate_parent_string + ' output the most likely parent class for the entity given by user. \n'

        prompt += '<|user|>\n Find the parent for ' + child + '\n'

        prompt += '<|assistant|>\n'

        user_prompt = prompt

        prompt += 'The parent class is ' + parent

        return prompt, user_prompt

    def Find_Child_prompter(parent, children):
            
        children_string = "{" + ", ".join(children) + "}"

        prompt = '<|system|>\nYou are a helpful assistant. Given a category, output entities that belong to this category \n'

        prompt += '<|user|>\n' + 'Find other entities belonging to category ' + parent + "\n"

        prompt += '<|assistant|>\n'

        user_prompt = prompt

        prompt += 'The expanded entities belonging to the category ' + parent + ' are ' + children_string

        return prompt, user_prompt

    def generate_and_tokenize_prompt_TaxoExpan(data_point):
        
        child = data_point['child']
        parent = data_point['parent']
        candidate_parents = data_point['candidate_parents']

        prompt, user_prompt =  Taxonomy_Expansion_prompter(child, parent, candidate_parents)
            
        tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
        )

        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt = tokenize(prompt)

        tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably

        return tokenized_full_prompt

    def generate_and_tokenize_prompt_SetExpan(data_point):

        input_entities = data_point['input']
        output_entities = data_point['output']
        
        prompt, user_prompt = Set_Expansion_prompter(input_entities, output_entities)
            
        tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
        )

        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        

        tokenized_full_prompt = tokenize(prompt)

        tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably

        return tokenized_full_prompt

    def generate_and_tokenize_prompt_SetExpan_parents(data_point):

        input_entities = data_point['input']
        output_entities = data_point['output']
        parent = data_point['parent']

        prompt, user_prompt = Set_Expansion_parent_prompter(input_entities, output_entities, parent)
            
        tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
        )

        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        

        tokenized_full_prompt = tokenize(prompt)

        tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably

        return tokenized_full_prompt
    
    def generate_and_tokenize_prompt_Find_Child(data_point):
            
        parent = data_point['parent']
        children = data_point['children']

        prompt, user_prompt = Find_Child_prompter(parent, children)
            
        tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
        )

        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        
        tokenized_full_prompt = tokenize(prompt)

        tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        
        return tokenized_full_prompt

    def generate_and_tokenize_prompt(data_point):

        if data_point['task'] == "Set-Expan":
            if set_expan_parent:
                return generate_and_tokenize_prompt_SetExpan_parents(data_point)
            else:
                return generate_and_tokenize_prompt_SetExpan(data_point)
        if data_point['task'] == "Taxo-Expan":
            return generate_and_tokenize_prompt_TaxoExpan(data_point)
        else:
            return generate_and_tokenize_prompt_Find_Child(data_point)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )

        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    # For int8
    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
        
    data = load_dataset("json", data_files=data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    print(len(train_data), " lines of training data loaded ")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            # evaluation_strategy="steps" if val_set_size > 0 else "no",
            # eval_steps=200 if val_set_size > 0 else None,
            # save_steps=200,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_set_size > 0 else "no",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            gradient_checkpointing=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
