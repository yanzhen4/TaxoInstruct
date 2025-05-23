export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/shared/data3/yanzhen4/TaxoInstruct/models_cache # Replace with your cache path
export HUGGINGFACE_HUB_CACHE=/shared/data3/yanzhen4/TaxoInstruct/models_cache # Replace with your cache path
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

dataset='APR'

folder="/shared/data3/yanzhen4/TaxoInstruct_ACL" # Replace with the path to your folder

seed=3408
run_name=$seed
lora_weights="$folder/model/Pretrain_CTD_7e-5"
find_parent_input_data="$folder/datasets/SetExpan/$dataset.jsonl"
find_parent_input_prompt_data="$folder/intermediate/find_parent_prompt_$dataset.jsonl"
find_parent_output_path="$folder/intermediate/find_parent_new_$dataset.jsonl"
find_parent_phrase_output_path="$folder/output/SetExpan/$dataset/$run_name"
setexpan_input_data="$folder/intermediate/setexpan_input_$dataset.jsonl"
setexpan_input_prompt_data="$folder/intermediate/setexpan_input_prompt_$dataset.jsonl"
setexpan_output_data="$folder/intermediate/setexpan_output_findParent_$dataset.jsonl"
insufficient_data_path="$folder/intermediate/supplement/$dataset"
insufficient_data_output_path="$folder/intermediate/supplement/setexpan_output_findParent_$dataset.jsonl"

python3 $folder/preprocess_data.py \
    --input_data_path  $find_parent_input_data\
    --output_data_path  $find_parent_input_prompt_data\

python3 $folder/inference.py \
    --test_data_path $find_parent_input_prompt_data \
    --lora_weights $lora_weights \
    --output_path $find_parent_output_path \
    --seed $seed

python3 $folder/output/SetExpan/phrase_parent.py \
    --dataset $dataset\
    --input_data_path $find_parent_output_path\
    --output_data_path $find_parent_phrase_output_path

python3 $folder/output/SetExpan/preprocess_SetExpan_data.py \
    --parent_data_path $find_parent_phrase_output_path\
    --setexpan_input_data_path $find_parent_input_data\
    --output_data_path $setexpan_input_data\

python3 $folder/preprocess_data.py \
    --input_data_path  $setexpan_input_data\
    --output_data_path  $setexpan_input_prompt_data\

python3 $folder/inference.py \
    --test_data_path $setexpan_input_prompt_data \
    --lora_weights $lora_weights \
    --output_path $setexpan_output_data \
    --seed $seed

python3 $folder/output/SetExpan/phrase_children.py \
    --original_data_path $find_parent_input_data\
    --input_data_path $setexpan_output_data\
    --output_data_path $folder/output/SetExpan/$dataset/$run_name/first_run \
    --insufficient_data_path $insufficient_data_path \
    --parent_data_path  $find_parent_phrase_output_path \
    --run_name $run_name\
    --supplement_query True \

python3 $folder/preprocess_data.py \
    --input_data_path  $insufficient_data_path/supplement_query.jsonl\
    --output_data_path  $insufficient_data_path/supplement_query_prompt.jsonl\

python3 $folder/inference.py \
    --test_data_path $insufficient_data_path/supplement_query_prompt.jsonl \
    --lora_weights $lora_weights \
    --output_path $insufficient_data_output_path \
    --seed $seed

python3 $folder/output/SetExpan/phrase_children.py \
    --original_data_path $insufficient_data_path/supplement_query.jsonl\
    --input_data_path $insufficient_data_output_path\
    --output_data_path $folder/output/SetExpan/$dataset/$run_name/supplement \
    --insufficient_data_path $insufficient_data_path \
    --run_name $run_name\
    --supplement_query False

python3 $folder/output/SetExpan/concat_rank_supplement.py \
    --dataset $dataset\
    --setexpan_output_folder $folder/output/SetExpan/$dataset/$run_name\