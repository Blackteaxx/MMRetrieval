deepspeed --include localhost:0,1 run_embedding.py \
    --output_dir modeloutput/1221/Triple \
    --model_name_or_path model/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --data_dir data/train.json \
    --img_dir data/train_images \
    --cache_dir cache_data \
    --temperature 0.05 \
    --loss_type triplet \
    --negatives_cross_batch True \
    --deepspeed ds_zero2_no_offload.json \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.15 \
    --dataloader_num_workers 16 \
    --fp16 True \
    --num_train_epochs 15 \
    --per_device_train_batch_size 8 \
    --remove_unused_columns False \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --temperature 0.05 \
    --logging_steps 5 \


