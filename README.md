# 项目说明

本项目旨在处理和分析图像与文本数据，特别是在电子商务领域的应用。项目包含数据处理、模型训练、推理和评估等多个模块。

数据集主要包括 Coupert 和 Shopee 两个数据集，用于多模态产品匹配任务，核心要点是将图像文本数据融合映射到一个好的嵌入空间。

同时还有一份用于推理的代码，在 `gpu179, user: hutu, path: ~/workspace/codes/` 下，相关说明参考目录下的 `README.md`。

尝试的模型包括 BGE, Visual BGE, CLIP, Marqo FashionSigLIP。其中 BGE与CLIP尝试进行微调，其余均采用了 zero-shot 方式。

更多细节参考 相关资源下的 [参考文档] 和 [项目文档]。

> 项目构建时还处于学习阶段，部分代码可能存在问题，仅供参考。

## 目录结构

```
.
├── data/                                # 数据目录
│   ├── all_hw_images_cleaned.parquet    # 购物党海外数据
│   ├── coupert/                         # Coupert 数据集
│   │   ├── arc/                         # 归档数据
│   │   │   ├── coupert.jsonl            # JSONL 格式数据
│   │   │   └── coupert_raw.tsv          # TSV 格式原始数据
│   │   ├── coupert_eval.jsonl           # 评估数据
│   │   ├── coupert_eval_retrieval.jsonl # 检索评估数据
│   │   ├── coupert_gallery.jsonl        # 图库数据
│   │   ├── coupert_train.jsonl          # 训练数据
│   │   ├── process_data.ipynb           # 数据处理 Notebook
│   │   └── prs.py                       # 数据处理脚本
│   └── shopee/                          # Shopee 相关数据集
│       ├── hn_mine.json                 # BGE hard nagetives 数据，用于训练
│       ├── marqo.json                   # Marqo 相关数据数据
│       ├── process_data.ipynb           # 数据处理 Notebook
│       ├── sample_submission.csv        # 示例提交文件
│       ├── shopee-product-matching.zip  # Shopee 产品匹配比赛数据压缩包
│       ├── test.csv                     # 比赛原始测试数据
│       ├── train.csv                    # 比赛原始训练数据
│       └── train.json                   # 训练数据（JSON 格式）
├── ds_zero2_no_offload.json             # DeepSpeed 配置文件
├── hn_mine.sh                           # hard negatives挖掘脚本
├── log.log                              # 日志文件
├── modeloutput/                         # 模型输出目录
│   └── FlagEmbedding/                   # FlagEmbedding 模型输出
│       └── runs/                        # 训练运行记录
│           ├── Dec13_14-02-59_gpu179/   # 运行记录 1
│           ├── Dec13_14-04-54_gpu179/   # 运行记录 2
│           └── Dec13_14-06-11_gpu179/   # 运行记录 3
├── notebook/                            # Notebook 目录
│   ├── clip_infer.py                    # Coupert CLIP 推理脚本
│   ├── coupert_infer_bge.ipynb          # Coupert BGE 推理 Notebook
│   ├── coupert_infer_clip.ipynb         # Coupert CLIP 推理 Notebook
│   ├── coupert_infer_marqo.ipynb        # Coupert Marqo 推理 Notebook
│   ├── coupert_infer_reranker.ipynb     # Coupert 重排序推理 Notebook
│   ├── coupert_infer_retrieval.ipynb    # Coupert 检索推理 Notebook
│   ├── marqo_ddp.py                     # Coupert Marqo 分布式推理脚本
│   ├── marqo_infer.py                   # Coupert Marqo 推理脚本
│   ├── model/                           # Shopee 比赛NFNet训练模型文件目录
│   │   ├── model_0_24.35597083634365.pt # 模型文件 0
│   │   ├── model_1_21.790381922410656.pt # 模型文件 1
│   │   └── ...                          # 其他模型文件
│   ├── qwen_vl_rerank_test.ipynb        # Qwen-VL 重排序测试 Notebook
│   ├── shopee_infer_2.ipynb             # Shopee 推理 Notebook 2
│   ├── shopee_infer.ipynb               # Shopee 推理 Notebook
│   ├── shopee_infer_rerank.ipynb        # Shopee 重排序推理 Notebook
│   ├── shopee-marqo-infer.ipynb         # Shopee Marqo 推理 Notebook
│   ├── shopee-nfnet-infer.ipynb         # Shopee NFNet 推理 Notebook
│   └── shopee-nfnet-train.ipynb         # Shopee NFNet 训练 Notebook
├── README.md                            # 项目说明文件
├── run_embedding.py                     # 模型训练脚本
├── run_embedding.sh                     # 模型训练脚本（Shell）
├── src/                                 # 源代码目录
│   ├── arguments.py                     # 参数配置脚本
│   ├── criterion.py                     # 损失函数脚本
│   ├── dataset/                         # 数据集处理目录
│   │   ├── coupert.py                   # Coupert 数据集处理脚本
│   │   ├── __init__.py                  # 初始化文件
│   │   └── shopee.py                    # Shopee 数据集处理脚本
│   ├── __init__.py                      # 初始化文件
│   ├── modeling/                        # 模型定义目录
│   │   ├── marqo_fashionSigLIP.py       # Marqo FashionSigLIP 模型
│   │   ├── modeling_clip.py             # CLIP 模型定义
│   │   ├── modeling_visbge.py           # Visual BGE 模型定义
│   │   └── visual_bge/                  # Visual BGE 模型目录( Copied from FlagEmbedding's Visual BGE)
│   └── training/                        # 训练脚本目录
│       └── trainer.py                   # 训练器脚本
├── utils/                                # 工具脚本目录
│   ├── embedder.py                       # 嵌入生成工具
│   ├── metrics.py                        # 评估指标工具
│   ├── reranker.py                       # 重排序工具
│   ├── retrieval.py                      # 检索工具
│   └── visualize.py                      # 可视化工具
```

## 数据集

1. shopee-product-matching: Shopee 产品（Image & Text）匹配数据集，包含训练和测试数据。目录为 `data/shopee/`。
    - 大部分测试脚本都在 `notebook/shopee*.ipynb` 中。
    - 训练脚本在 `run_embedding.sh` 中，使用 transformers Trainer进行训练，支持deepspeed多卡训练。

2. Coupert: Coupert 产品（Image & Text）匹配数据集，包含训练、评估和检索数据。目录为 `data/coupert/`。
    - 大部分测试脚本都在 `notebook/coupert*.ipynb` 中。
    - 训练脚本在 `run_embedding.sh` 中，使用 transformers Trainer进行训练，支持deepspeed多卡训练。

### 其余可用的数据集

- Product1M：中文，化妆品，可做Instance-level Retrieval
- M5Product：中文，可做Fine-grained Retrieval，但是需要自定义召回样本
- eSSPR：英文，可做Fine-grained Retrieval，数据集格式不明确，测试集4500，且仅有单个召回样本
- Mep-3M: 中文，可做Coarse-grained Retrieval

## 采用方法

目前效果最好的方法是 Marqo 在商品数据集上大规模预训练的FashionSigLIP模型，其次是使用CLIP 进行召回阶段的对比学习微调。

CLIP微调结构可以参考 `src/modeling/modeling_clip.py`，训练损失函数为 `src/criterion.py` 中的两个可用损失

- `TripletLogitLoss`: SigLIP-like 损失函数
- `InfoNCELoss`: 常用对比学习损失函数

## 调试方法

### 数据处理

需要注意的是，Coupert 数据集的使用需要先处理数据，但是我的处理方式为 query & gallery 的方式，与mjy的处理方式不同，所以需要注意数据的处理方式。

### 训练

如果想要了解模型的训练过程，可以启动 `run_embedding.sh` 脚本，在 `run_embedding.py` 脚本前加上 `debugpy` 调试模块，然后在vscode中设置`launch.json`进行调试，具体教程参考下面的链接。

[如何优雅的在VSCode中调试Python代码](https://github.com/yuanzhoulvpi2017/vscode_debug_transformers)

### 推理

推理过程中，可以参考使用 `notebook/*.ipynb` 中的代码进行推理，其中包含了模型加载、数据处理、推理和评估等步骤。

需要注意，每一个notebook都是在项目根目录底下运行之后，随后归档放入notebook目录下的，所以需要注意路径问题，其余无需修改。

## 使用方法

### 数据处理
1. 使用 `data/coupert/process_data.ipynb` 处理 Coupert 数据集。
2. 使用 `data/shopee/process_data.ipynb` 处理 Shopee 数据集。

### 模型训练
1. 使用 `src/training/trainer.py` 进行模型训练脚本搭建。
2. 使用 `run_embedding.sh` 进行模型训练。(注意修改参数)

### 模型推理
1. 使用 `notebook/coupert_infer_bge.ipynb` 进行 Coupert BGE 推理。
2. 使用 `notebook/shopee_infer.ipynb` 进行 Shopee 推理。

### 评估
1. 使用 `notebook/coupert_infer_retrieval.ipynb` 进行检索评估。
2. 使用 `notebook/shopee_infer_rerank.ipynb` 进行重排序评估。

## 依赖

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- DeepSpeed

> 可以与 `product-on-sale-recommendation` 项目共用一个环境。

## 相关资源

- MJY 的 Coupert 数据路径: gpu 176, path: `/ssd1/embedding_data/coupert_dataset/`
- [Shopee 项目文档](https://hno041oake.feishu.cn/wiki/PUGRwcc0SiXsq8kwRUJc0ewpnpb)
- [Coupert 项目文档](https://hno041oake.feishu.cn/wiki/F22gwAidHiMlWnkpXOwcjS2RnOh)