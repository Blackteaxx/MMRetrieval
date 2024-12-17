CUDA_VISIBLE_DEVICES=1 python run_embedding.py \
    --output_dir modeloutput/1211/infoNCE \
    --model_name_or_path model/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --data_dir data/train.json \
    --img_dir data/train_images \
    --cache_dir cache_data \
    --mode train \
    --learning_rate 2e-5 \
    --dataloader_num_workers 16 \
    --fp16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 256 \
    --remove_unused_columns False \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --temperature 0.05 \
    --logging_steps 5 \
    --warmup_steps 100 \

