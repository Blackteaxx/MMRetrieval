export CUDA_VISIBLE_DEVICES=5,6,7

input_file="data/train.json"
output_file="data/hn_mine.json"
model_path="/data/model/BAAI/bge-large-en-v1.5"
script_path="FlagEmbedding/scripts/hn_mine.py"

python $script_path \
    --input_file $input_file \
    --output_file $output_file \
    --range_for_sampling 50-200 \
    --negative_number 15 \
    --embedder_name_or_path $model_path \
    --embedder_model_class "encoder-only-base" \
    --normalize_embeddings \