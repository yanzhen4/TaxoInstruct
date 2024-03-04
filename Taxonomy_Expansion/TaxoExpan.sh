export CUDA_VISIBLE_DEVICES=0,2

pretrain_weight_path="Pretraining/CTD_Pretrain_weights"
finetune_training_data="Taxonomy_Expansion/train_Taxo_Expan_environment.jsonl"
finetune_weight_path="Taxonomy_Expansion/finetune_weights/environment_finetune"

taxo_expan_data_input_path="Taxonomy_Expansion/test_Taxo_Expan_environment.jsonl"
taxo_expan_data_output_path="Taxonomy_Expansion/specter/test_Taxo_Expan_environment.jsonl"

#Finetune model on Environment/Science
WORLD_SIZE=2  torchrun --nproc_per_node=2 --master_port=21420 finetune_llama.py \
    --base_model "/shared/data3/checkpoints/llama-2-7b-chat-hf" \
    --num_epochs 5 \
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

python3 inference_llama.py \
    --base_model "/shared/data3/checkpoints/llama-2-7b-chat-hf" \
    --lora_weights $finetune_weight_path \
    --output_name  $taxo_expan_data_output_path\
    --data_path  $taxo_expan_data_input_path\
    --full_parent False \
    --complete_output False\
    --set_expan_parent True

