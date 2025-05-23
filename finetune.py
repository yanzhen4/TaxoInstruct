import wandb
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import fire

def train_llama3(
    train_data_path: str = None,
    eval_data_path: str = None,
    project_name: str = "train_llama3_0720",
    run_name: str = 'train_llama',
    base_model: str = "unsloth/llama-3-8b-bnb-4bit",
    lora_weights: str = None,
    num_train_epochs: int = 5,
    lr: float = 2e-4,
    seed: int = 3407,
    save_total: int = 1, 
    eval_steps: int = 100
    ):
    
    print("train_data_path: ", train_data_path)
    print("eval_data_path: ", eval_data_path)
    print("base_model: ", base_model)
    print("lora_weights: ", lora_weights)
    print("run_name: ", run_name)
    print("num_train_epochs: ", num_train_epochs)
    print("lr: ", lr)
    print("seed: ", seed)
    print("save_total: ", save_total)
    print("eval_steps: ", eval_steps)
    print("project_name: ", project_name)

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    wandb.init(project=project_name, name=run_name)

    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
    ] # More models at https://huggingface.co/unsloth

    if lora_weights:

        print("Load from existing weights")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = lora_weights,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    else:

        print("Load from original model")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = base_model,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            device_map="auto"
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","embed_tokens","lm_head"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        tasks   = examples["task"]
        inputs  = examples["input"]
        outputs = examples["output"]

        texts = []
        for task, input, output in zip(tasks, inputs, outputs):

            if task == 'Set-Expan':
                instruction = "Given a category and an entity set belonging to this category, output other entities belonging to this category and sharing the same granularity as the seeds."
            elif task == 'Taxo-Expan':
                instruction = "Given a set of candidate parent classes and an entity, output the most likely parent class for the entity given by user."
            elif task == 'Find-Parent':
                instruction = "Given a set of entities, output the most likely parent class for the entities given by the user"
            else:
                raise ValueError(f"Unknown task: {task}")

            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN

            texts.append(text)

        return { "text" : texts, }

    train_dataset = load_dataset("json", data_files=train_data_path)
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)['train']
    train_dataset = train_dataset.shuffle(seed=3407)

    if eval_data_path:
        eval_dataset = load_dataset("json", data_files=eval_data_path)
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)['train']
    else:
        eval_dataset = None

    effective_batch_size = 2 * 4  # per_device_train_batch_size * gradient_accumulation_steps
    total_steps = (len(train_dataset) // effective_batch_size) * num_train_epochs
    save_steps = total_steps // save_total

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = num_train_epochs,
            learning_rate = lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = seed,
            output_dir = f"outputs/{run_name}",
            evaluation_strategy = "steps",
            save_steps = save_steps,
            eval_steps = eval_steps,
        ),
    )

    trainer_stats = trainer.train()

    model.save_pretrained(f"outputs/{run_name}")
    tokenizer.save_pretrained(f"outputs/{run_name}")

    print(trainer_stats)

if __name__ == "__main__":
    fire.Fire(train_llama3)