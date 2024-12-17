export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1

train_data="/home/hutu/workspace/MMRetrieval/data/hn_mine.json"
model_path="/home/hutu/workspace/model/bge_base_en_v1.5"
deepspeed_config="/home/hutu/workspace/MMRetrieval/FlagEmbedding/examples/finetune/ds_stage0.json"

# set large epochs and small batch size for testing
num_train_epochs=4
per_device_train_batch_size=1

# set num_gpus to 2 for testing
num_gpus=2

# Set TORCH_CUDA_ARCH_LIST and allow unsupported compiler
# export TORCH_CUDA_ARCH_LIST="7.5"
# export CXXFLAGS="-allow-unsupported-compiler"

model_args="\
    --model_name_or_path $model_path \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
"

training_args="\
    --output_dir modeloutput/FlagEmbedding \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
