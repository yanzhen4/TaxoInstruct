export CUDA_VISIBLE_DEVICES=6
# export TRANSFORMERS_CACHE=/path/to/cache

pretrain_weight_path="Pretraining/CTD_Pretrain_weights"

set_expan_data_path="Entity_Set_Expansion/Set_Expan_input.jsonl"
set_expan_output_path="Entity_Set_Expansion/Set_Expan_output.jsonl"

python3 inference_llama.py \
     --base_model "/shared/data3/checkpoints/llama-2-7b-chat-hf" \
     --lora_weights $pretrain_weight_path \
     --output_name  $set_expan_output_path\
     --data_path  $set_expan_data_path\
     --load_weights True\
     --full_parent False \
     --complete_output True\
     --set_expan_parent True