# 项目说明

本项目旨在处理和分析图像与文本数据，特别是在电子商务领域的应用。项目包含数据处理、模型训练、推理和评估等多个模块。以下是项目的详细说明和目录结构。

## 目录结构

```
.
├── data/                                # 数据目录
│   ├── all_hw_images_cleaned.parquet    # 清洗后的图像数据
│   ├── coupert/                         # Coupert 数据集
│   │   ├── arc/                         # 原始数据
│   │   │   ├── coupert.jsonl            # JSONL 格式数据
│   │   │   └── coupert_raw.tsv          # TSV 格式原始数据
│   │   ├── coupert_eval.jsonl           # 评估数据
│   │   ├── coupert_eval_retrieval.jsonl # 检索评估数据
│   │   ├── coupert_gallery.jsonl        # 图库数据
│   │   ├── coupert_train.jsonl          # 训练数据
│   │   ├── process_data.ipynb           # 数据处理 Notebook
│   │   └── prs.py                       # 数据处理脚本
│   └── shopee/                          # Shopee 数据集
│       ├── hn_mine.json                 # 热门挖掘数据
│       ├── marqo.json                   # Marqo 数据
│       ├── process_data.ipynb           # 数据处理 Notebook
│       ├── sample_submission.csv        # 示例提交文件
│       ├── shopee-product-matching.zip  # Shopee 产品匹配数据
│       ├── test.csv                     # 测试数据
│       ├── train.csv                    # 训练数据
│       └── train.json                   # 训练数据（JSON 格式）
├── ds_zero2_no_offload.json             # DeepSpeed 配置文件
├── hn_mine.sh                           # 热门挖掘脚本
├── log.log                              # 日志文件
├── modeloutput/                         # 模型输出目录
│   └── FlagEmbedding/                   # FlagEmbedding 模型输出
│       └── runs/                        # 训练运行记录
│           ├── Dec13_14-02-59_gpu179/   # 运行记录 1
│           ├── Dec13_14-04-54_gpu179/   # 运行记录 2
│           └── Dec13_14-06-11_gpu179/   # 运行记录 3
├── notebook/                            # Notebook 目录
│   ├── clip_infer.py                    # CLIP 推理脚本
│   ├── coupert_infer_bge.ipynb          # Coupert BGE 推理 Notebook
│   ├── coupert_infer_clip.ipynb         # Coupert CLIP 推理 Notebook
│   ├── coupert_infer_marqo.ipynb        # Coupert Marqo 推理 Notebook
│   ├── coupert_infer_reranker.ipynb     # Coupert 重排序推理 Notebook
│   ├── coupert_infer_retrieval.ipynb    # Coupert 检索推理 Notebook
│   ├── marqo_ddp.py                     # Marqo 分布式训练脚本
│   ├── marqo_infer.py                   # Marqo 推理脚本
│   ├── model/                           # 模型文件目录
│   │   ├── model_0_24.35597083634365.pt # 模型文件 0
│   │   ├── model_1_21.790381922410656.pt # 模型文件 1
│   │   └── ...                          # 其他模型文件
│   ├── __pycache__/                     # Python 编译缓存
│   ├── qwen_vl_rerank_test.ipynb        # Qwen-VL 重排序测试 Notebook
│   ├── shopee_infer_2.ipynb             # Shopee 推理 Notebook 2
│   ├── shopee_infer.ipynb               # Shopee 推理 Notebook
│   ├── shopee_infer_rerank.ipynb        # Shopee 重排序推理 Notebook
│   ├── shopee-marqo-infer.ipynb         # Shopee Marqo 推理 Notebook
│   ├── shopee-nfnet-infer.ipynb         # Shopee NFNet 推理 Notebook
│   └── shopee-nfnet-train.ipynb         # Shopee NFNet 训练 Notebook
├── README.md                            # 项目说明文件
├── run_embedding.py                     # 嵌入生成脚本
├── run_embedding.sh                     # 嵌入生成脚本（Shell）
├── src/                                 # 源代码目录
│   ├── arguments.py                     # 参数配置脚本
│   ├── criterion.py                     # 损失函数脚本
│   ├── dataset/                         # 数据集处理目录
│   │   ├── coupert.py                   # Coupert 数据集处理脚本
│   │   ├── __init__.py                  # 初始化文件
│   │   ├── __pycache__/                 # Python 编译缓存
│   │   └── shopee.py                    # Shopee 数据集处理脚本
│   ├── __init__.py                      # 初始化文件
│   ├── modeling/                        # 模型定义目录
│   │   ├── marqo_fashionSigLIP.py       # Marqo FashionSigLIP 模型
│   │   ├── modeling_clip.py             # CLIP 模型定义
│   │   ├── modeling_visbge.py           # Visual BGE 模型定义
│   │   ├── __pycache__/                 # Python 编译缓存
│   │   └── visual_bge/                  # Visual BGE 模型目录
│   │       ├── eva_clip/                # EVA CLIP 模型
│   │       │   ├── bpe_simple_vocab_16e6.txt.gz # BPE 词汇表
│   │       │   ├── constants.py         # 常量定义
│   │       │   ├── eva_vit_model.py     # EVA ViT 模型
│   │       │   ├── factory.py           # 工厂模式
│   │       │   ├── hf_configs.py        # HuggingFace 配置
│   │       │   ├── hf_model.py          # HuggingFace 模型
│   │       │   ├── __init__.py          # 初始化文件
│   │       │   ├── loss.py              # 损失函数
│   │       │   ├── model_configs/       # 模型配置文件
│   │       │   ├── model.py             # 模型定义
│   │       │   ├── modified_resnet.py   # 修改后的 ResNet
│   │       │   ├── openai.py            # OpenAI 模型
│   │       │   ├── pretrained.py        # 预训练模型
│   │       │   ├── __pycache__/         # Python 编译缓存
│   │       │   ├── rope.py              # RoPE 实现
│   │       │   ├── timm_model.py        # TIMM 模型
│   │       │   ├── tokenizer.py         # 分词器
│   │       │   ├── transformer.py       # Transformer 模型
│   │       │   ├── transform.py         # 数据转换
│   │       │   └── utils.py             # 工具函数
│   │       └── modeling.py              # Visual BGE 模型定义
│   ├── __pycache__/                     # Python 编译缓存
│   └── training/                        # 训练脚本目录
│       └── trainer.py                   # 训练器脚本
├── utils/                                # 工具脚本目录
│   ├── embedder.py                       # 嵌入生成工具
│   ├── metrics.py                        # 评估指标工具
│   ├── __pycache__/                      # Python 编译缓存
│   ├── reranker.py                       # 重排序工具
│   ├── retrieval.py                      # 检索工具
│   └── visualize.py                      # 可视化工具
```

## 使用方法

### 数据处理
1. 使用 `data/coupert/process_data.ipynb` 处理 Coupert 数据集。
2. 使用 `data/shopee/process_data.ipynb` 处理 Shopee 数据集。

### 模型训练
1. 使用 `src/training/trainer.py` 进行模型训练。
2. 使用 `notebook/shopee-nfnet-train.ipynb` 进行 NFNet 模型训练。

### 模型推理
1. 使用 `notebook/coupert_infer_bge.ipynb` 进行 Coupert BGE 推理。
2. 使用 `notebook/shopee_infer.ipynb` 进行 Shopee 推理。

### 评估
1. 使用 `notebook/coupert_infer_retrieval.ipynb` 进行检索评估。
2. 使用 `notebook/shopee_infer_rerank.ipynb` 进行重排序评估。

## 依赖

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- DeepSpeed
