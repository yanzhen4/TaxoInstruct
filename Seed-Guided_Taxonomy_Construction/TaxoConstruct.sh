export CUDA_VISIBLE_DEVICES=1,3
# export TRANSFORMERS_CACHE=/path/to/cache

pretrain_weight_path="Pretraining/CTD_Pretrain_weights"
finetune_training_data="Pubmed_CVD/finetune_TaxoConstruct.jsonl"
finetune_weight_path="Seed-Guided_Taxonomy_Construction/finetune_weights/CVD_finetune"

#Train model on Pubmed_CVD given structure
WORLD_SIZE=2  torchrun --nproc_per_node=2 --master_port=21419 finetune_llama2.py \
    --base_model "/shared/data3/checkpoints/llama-2-7b-chat-hf" \
    --num_epochs 10 \
    --cutoff_len 2048 \
    --data_path $finetune_training_data \
    --output_dir $finetune_weight_path \
    --lora_target_modules ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","embed_tokens","lm_head"] \
    --lora_r 16 \
    --micro_batch_size 4 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --val_set_size 0 \
    --train_on_inputs False \
    --resume_from_checkpoint $pretrain_weight_path

set_expan_data_path="Seed-Guided_Taxonomy_Construction/Set_Expan_Expand_entities.jsonl"
set_expan_output_path="Seed-Guided_Taxonomy_Construction/Set_Expan_Expand_entities_output.jsonl"

python3 inference_llama.py \
     --base_model "/shared/data3/checkpoints/llama-2-7b-chat-hf" \
     --lora_weights $finetune_weight_path \
     --output_name  $set_expan_output_path\
     --data_path  $set_expan_data_path\
     --full_parent False \
     --complete_output False\
     --set_expan_parent True

