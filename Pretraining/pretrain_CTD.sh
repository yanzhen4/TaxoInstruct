export CUDA_VISIBLE_DEVICES=0,1
# export TRANSFORMERS_CACHE=/path/to/cache

training_data="Pretraining/CTD_Pretrain.jsonl"
output_folder="Pretraining/CTD_Pretrain_weights"

WORLD_SIZE=2  torchrun --nproc_per_node=2 --master_port=21234 finetune_llama2.py \
    --base_model "/shared/data3/checkpoints/llama-2-7b-chat-hf" \
    --num_epochs 15 \
    --cutoff_len 2048 \
    --data_path $training_data \
    --output_dir $output_folder \
    --lora_target_modules ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","embed_tokens","lm_head"] \
    --lora_r 16 \
    --micro_batch_size 4 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --val_set_size 0 \
    --train_on_inputs False \
    --set_expan_parent True
