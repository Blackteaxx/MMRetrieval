export CUDA_VISIBLE_DEVICES=1

input_file="/home/hutu/workspace/MMRetrieval/data/train.json"
output_file="/home/hutu/workspace/MMRetrieval/data/hn_mine.json"
model_path="/home/hutu/workspace/SentenceEmbedding/modeloutput/1214/checkpoint-2142"
script_path="/home/hutu/workspace/MMRetrieval/FlagEmbedding/scripts/hn_mine.py"

python $script_path \
    --input_file $input_file \
    --output_file $output_file \
    --range_for_sampling 50-200 \
    --negative_number 15 \
    --embedder_name_or_path $model_path \
    --embedder_model_class "encoder-only-base" \
    --normalize_embeddings \