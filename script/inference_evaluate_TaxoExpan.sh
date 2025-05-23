export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/shared/data3/yanzhen4/TaxoInstruct/models_cache # Replace with your cache path
export HUGGINGFACE_HUB_CACHE=/shared/data3/yanzhen4/TaxoInstruct/models_cache # Replace with your cache path
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

dataset="science"

folder="/shared/data3/yanzhen4/TaxoInstruct_ACL" # Replace with the path to your folder

lora_weights="$folder/model/TaxoExpan/finetune_$dataset"
input_data_path="$folder/datasets/TaxoExpan/test_TaxoExpan_20_$dataset.jsonl"
input_data_prompt_path="$folder/intermediate/test_TaxoExpan_$dataset.jsonl"
inference_output_path="$folder/output/TaxoExpan/output_$dataset.jsonl"
scores_output_path="$folder/output/TaxoExpan/score_new.txt"
raw_data_path="$folder/datasets/TaxoExpan/raw_data"

python3 $folder/preprocess_data.py \
    --input_data_path  $input_data_path\
    --output_data_path  $input_data_prompt_path\

python3 $folder/inference.py \
    --test_data_path $input_data_prompt_path \
    --lora_weights $lora_weights \
    --output_path $inference_output_path \

python3 $folder/output/$task/evaluate.py \
    --output_path $inference_output_path \
    --scores_path $scores_output_path   \
    --raw_data_path $raw_data_path \
    --dataset $dataset