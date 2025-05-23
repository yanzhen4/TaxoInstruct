export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/shared/data3/yanzhen4/TaxoInstruct/models_cache # Replace with your cache path
export HUGGINGFACE_HUB_CACHE=/shared/data3/yanzhen4/TaxoInstruct/models_cache # Replace with your cache path
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

dataset="cvd"

folder="/shared/data3/yanzhen4/TaxoInstruct_ACL" # Replace with the path to your folder

seed=3408
run_name=$seed
SetExpan_lora_weights="$folder/model/Pretrain_CTD_7e-5"
TaxoExpan_lora_weights="$folder/model/HiExpan/finetune_$dataset"
SetExpan_input_data_path="$folder/datasets/HiExpan/$dataset/SetExpan_input_shuffle.jsonl"
SetExpan_input_data_prompt_path="$folder/intermediate/SetExpan_input_prompt_$run_name.jsonl"
SetExpan_output_path=$folder/intermediate/find_children_$dataset.jsonl
TaxoExpan_input_data_path=$folder/datasets/HiExpan/$dataset/TaxoExpan_input.jsonl
TaxoExpan_input_data_prompt_path=$folder/intermediate/TaxoExpan_input_prompt_$run_name.jsonl
TaxoExpan_output_data_path=$folder/intermediate/find_parent_$dataset.jsonl
TaxoExpan_phrase_output_data_path=$folder/output/HiExpan/$dataset/$run_name

python3 $folder/preprocess_data.py \
    --input_data_path  $SetExpan_input_data_path\
    --output_data_path  $SetExpan_input_data_prompt_path\

python3 $folder/inference.py \
    --test_data_path $SetExpan_input_data_prompt_path \
    --lora_weights $SetExpan_lora_weights \
    --output_path  $SetExpan_output_path\
    --seed $seed \

python3 $folder/output/HiExpan/generate_findParent_data.py \
    --dataset $dataset\
    --SetExpan_output_data_path $SetExpan_output_path\
    --TaxoExpan_input_data_path $TaxoExpan_input_data_path\

python3 $folder/preprocess_data.py \
    --input_data_path  $TaxoExpan_input_data_path\
    --output_data_path  $TaxoExpan_input_data_prompt_path\

python3 $folder/inference.py \
    --test_data_path $TaxoExpan_input_data_prompt_path \
    --lora_weights $TaxoExpan_lora_weights \
    --output_path  $TaxoExpan_output_data_path\
    --seed $seed\

python3 $folder/output/HiExpan/phrase_output_data.py \
    --dataset $dataset\
    --input_data_path $TaxoExpan_output_data_path\
    --output_data_path $TaxoExpan_phrase_output_data_path \
    --run_name $run_name 