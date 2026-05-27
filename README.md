# SigLIP Flickr8k 图文检索系统 / SigLIP Flickr8k Image-Text Retrieval System

<div align="center">

**基于 SigLIP (Sigmoid Loss for Language-Image Pretraining) 在 Flickr8k 上实现的小规模图文检索系统**

A small-scale image-text retrieval system based on SigLIP, trained and evaluated on Flickr8k

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.15+-orange.svg)](https://gradio.app/)

</div>

---

## 🌟 项目亮点 / Key Highlights

| 指标 / Metric | 值 / Value | 说明 / Description |
|---|---|---|
| **最优 t2i R@1** | **58.03%** | Exp21 双预训练 + 部分解冻 |
| **最优 i2t R@1** | **60.09%** | Exp21 同上 |
| **vs 基线提升** | **+36.77%** | Exp21 vs Exp0 (21.26%) |
| **BF16 推理加速** | **3.82x** | 精度损失 <0.02pp |
| **总实验数** | **22 组** | Exp0-Exp21 系统性探索 |

---

## 📖 项目背景 / Project Background

本项目是"媒体与认知"课程的大作业，要求基于 SigLIP 模型在 Flickr8k 数据集上实现图文检索系统。与 CLIP 使用 softmax 全局归一化不同，SigLIP 对每个图文对独立使用 sigmoid 判断是否匹配，对 batch size 更鲁棒。

项目从最简单的基线 (Exp0: t2i R@1=21.26%) 开始，经过 22 组系统性实验探索，最终达到 58.03% (Exp21)，提升 36.77 个百分点。

---

## 📁 项目结构 / Project Structure

```
repository/
├── app_inference.py              # Gradio 推理 Demo（BF16/FP32 自动切换）
├── README.md                     # 本文件
├── requirements.txt              # 依赖列表
├── experiment_report.html        # 实验报告（可浏览器打开）
│
├── siglip/                       # 核心 Python 包
│   ├── __init__.py               # 顶层导出（SigLIPLoss, models, datasets 等）
│   ├── data/
│   │   ├── data_loader.py        # Flickr8kDataset, DistilBERTDataset, build_transform
│   │   └── __init__.py
│   ├── loss/
│   │   ├── siglip_loss.py        # SigLIPLoss（sigmoid pairwise loss, 可学习 t/b）
│   │   └── __init__.py
│   ├── utils/
│   │   ├── utils.py              # SimpleTokenizer, AverageMeter, read_caption_file, set_seed
│   │   └── __init__.py
│   ├── models/                   # 4 个模型架构，从简单到复杂
│   │   ├── siglip_model.py       # Exp baseline: ResNet18 + Transformer/MLP
│   │   ├── pretrained_siglip.py  # Exp19: 冻结 ViT-B/16 + 可训练 Transformer
│   │   ├── dual_pretrained_siglip.py    # Exp20: ViT partial + 全解冻 DistilBERT
│   │   ├── dual_pretrained_v2_siglip.py # Exp21: 双方部分解冻（最优模型）
│   │   ├── resnet_custom.py      # 自定义 ResNet18 (可调 width)
│   │   ├── transformer_encoder.py # 2层 Transformer 文本编码器
│   │   └── __init__.py
│   ├── inference/
│   │   ├── model_loader.py       # InferenceEngine（预计算全库 embedding + 快速检索）
│   │   ├── config.py             # 自动 GPU/CPU 检测, BF16 配置
│   │   └── __init__.py
│   └── benchmark/
│   │   ├── utils.py              # load_model, measure_latency, evaluate_retrieval
│   │   └── __init__.py
│
├── scripts/                      # 训练/评估/可视化脚本（从项目根目录运行）
│   ├── train_siglip.py           # Exp baseline 训练 (ResNet18 + Transformer/MLP)
│   ├── train_siglip_pretrained.py          # Exp19 训练
│   ├── train_siglip_dual_pretrained.py     # Exp20 训练
│   ├── train_siglip_dual_pretrained_v2.py  # Exp21 训练（最优模型）
│   ├── eval_exp19.py             # Exp19 测试集评估
│   ├── eval_exp20.py             # Exp20 测试集评估
│   ├── eval_exp21.py             # Exp21 测试集评估
│   ├── viz_exp19.py              # Exp19 可视化（热力图 + Top-K + Bad Case）
│   ├── viz_exp20.py              # Exp20 可视化
│   ├── viz_exp21.py              # Exp21 可视化
│   ├── visualize_heatmap.py      # 跨模态相似度矩阵热力图
│   ├── visualize_topk.py         # Top-K 检索 + Bad Case 自动挖掘
│   ├── make_comparison*.py       # GT vs Top-5 对比图生成 (4个)
│   ├── benchmark_base.py         # FP32 基线 benchmark
│   ├── quantize_fp16.py          # BF16/FP16 半精度推理实验
│   ├── quantize_dynamic.py       # INT8 动态量化实验
│   ├── prune_layers.py           # 层剪枝实验
│   ├── run_exp5.sh / run_exp6.sh # 训练启动脚本
│   └── download_pretrained_distilbert.py # 下载 DistilBERT 本地缓存
│
├── visualization/                # 预训练 SigLIP 可视化 Demo (Gradio)
│   ├── app.py                    # 主程序 (http://localhost:7860)
│   ├── config.py                 # MODEL_NAME, DEVICE
│   ├── model/siglip_wrapper.py   # 预训练 SigLIP Wrapper
│   ├── data/flickr8k_loader.py   # HF/local 数据加载
│   └── viz/v1_similarity_matrix.py # Plotly 热力图生成
│
├── docs/original_assignment/     # 老师原始作业要求和指南
│   ├── README_original.md        # 原版 README (作业描述)
│   ├── README.pdf                # PDF 版
│   ├── project_guide.html        # 项目指南 HTML
│   ├── model_arch.png            # 架构示意图
│   ├── topk.png                  # Top-K 示例图
│   └── 算力平台运行指南.md        # 云平台使用指南
│
├── report/                       # LaTeX 最终报告
│   ├── final_report.tex          # 源文件
│   ├── final_report.pdf          # 编译后 PDF（正式提交版本）
│   └── 基于SigLIP的小规模图文检索系统 — 中期报告.pdf  # 中期报告
│
├── Flickr8k/                     # 数据集 (gitignore Images/)
│   ├── Images/                   # 8091 张 JPG 图像 (gitignored)
│   ├── captions.txt              # 全部 caption (CSV)
│   ├── test_captions.txt         # 测试集 (1000张, 4046行)
│   ├── train_captions.txt        # 训练集 (6000张, 32364行)
│   └── val_captions.txt          # 验证集 (1000张, 4045行)
│
├── outputs/                      # 训练结果
│   ├── exp0-9/                   # Exp0-9 checkpoints + logs (gitignored .pt)
│   ├── exp20_dual_pretrained/    # Exp20 checkpoint (gitignored)
│   ├── exp21_dual_pretrained_v2/ # Exp21 best + latest checkpoint (gitignored)
│   ├── benchmark_results/        # 11 个 JSON (FP32/BF16/FP16/INT8/剪枝)
│   └── *.log                     # 18 个训练日志
│
└── viz_outputs/                  # 可视化输出图片 (gitignored)
```

---

## 🔧 安装 / Installation

```bash
git clone https://github.com/yikai04/siglip-flickr8k-retrieval.git
cd repository
pip install -r requirements.txt

# 推理 Demo 需要 Gradio
pip install gradio>=4.15.0
```

**预训练权重**：模型需要 `google/siglip-base-patch16-224` 和 `distilbert-base-uncased`。首次训练时自动从 HuggingFace 下载，也可本地缓存：

```bash
python scripts/download_pretrained_distilbert.py
```

---

## 📊 数据准备 / Data Preparation

下载 [Flickr8k 数据集](https://illinois.nlplab.org/data/) 并放置在 `Flickr8k/` 目录：

```
Flickr8k/
├── Images/           # 8091 张 JPG 图像
├── captions.txt      # 全部 caption (CSV: image,caption)
├── test_captions.txt  # 测试集 caption
├── train_captions.txt # 训练集 caption
└── val_captions.txt   # 验证集 caption
```

---

## 🚀 使用 / Usage

### 训练 / Training

```bash
# Exp21（最优模型）：双预训练 + 部分解冻
python scripts/train_siglip_dual_pretrained_v2.py \
    --data-dir Flickr8k --epochs 50 --batch-size 32 \
    --output-dir outputs/exp21_dual_pretrained_v2

# Exp19：冻结预训练 ViT + 可训练 Transformer
python scripts/train_siglip_pretrained.py \
    --data-dir Flickr8k --epochs 100 \
    --output-dir outputs/exp19_pretrained_vit

# Exp baseline：从头训练 ResNet18 + Transformer
python scripts/train_siglip.py \
    --data-dir Flickr8k --epochs 100 --batch-size 64 \
    --text-encoder transformer --output-dir outputs/exp4_cosine_100ep
```

### 评估 / Evaluation

```bash
python scripts/eval_exp21.py \
    --checkpoint outputs/exp21_dual_pretrained_v2/best_siglip.pt \
    --data-dir Flickr8k
```

### 推理 Demo / Inference Demo

```bash
python app_inference.py
# 访问 http://localhost:7861
# GPU -> BF16 加速 (3.82x), CPU -> FP32 正常推理
```

**功能**：
- **文本搜图 (Text → Image)**：输入英文描述，检索 Top-K 相关图像 (Gallery + Score)
- **图片搜文 (Image → Text)**：上传图片，检索 Top-K 匹配描述 (表格 + Score)
- 启动时预计算全库 embedding，检索仅需一次 dot product (<50ms)
- 分离编码器调用：t2i 只跑 `encode_text()`，i2t 只跑 `encode_image()`

### 预训练 SigLIP 可视化 / Visualization Demo

```bash
python -m visualization.app
# 访问 http://localhost:7860
# 展示 google/siglip-base-patch16-224 预训练模型的图文相似度矩阵
```

### 推理优化 Benchmark / Inference Benchmark

```bash
python scripts/benchmark_base.py --ckpt-path <ckpt> --data-dir Flickr8k
python scripts/quantize_fp16.py --ckpt-path <ckpt> --data-dir Flickr8k
python scripts/quantize_dynamic.py --ckpt-path <ckpt> --data-dir Flickr8k
python scripts/prune_layers.py --ckpt-path <ckpt> --data-dir Flickr8k
```

---

## 📈 实验结果 / Performance

### 全部实验完整对比 / Full Experiment Results

| Exp | 核心变更 / Key Change | t2i R@1 | i2t R@1 | t2i R@5 | t2i R@10 | 评价 / Verdict |
|-----|----------------------|---------|---------|---------|----------|----------------|
| 0 | 常数LR基线 / Constant LR | 21.26 | 20.88 | 45.77 | 56.72 | ❌ 基线 |
| 1 | **Cosine Warmup** | 29.83 | 32.07 | 55.54 | 65.50 | ✅ **关键改进 +8.57** |
| 2 | RRC(0.5) | 12.58 | 13.59 | 31.27 | 43.08 | ❌ 裁剪过激 |
| 3 | RRC(0.75) | 27.01 | 29.76 | 51.01 | 60.55 | ❌ 不如Resize |
| 4 | Cosine 100ep w=32 | 34.78 | 37.93 | 57.07 | 65.40 | ✅ 前期最优 |
| 5 | w=48+CJ(0.2) | 35.86 | 38.51 | 58.45 | 66.58 | ✅ 配置确立 |
| **6** | **w=48+CJ+EMA** | **36.65** | **39.82** | 58.90 | 66.76 | ✅ **从头训练最优** |
| 7 | 4L+8H+d512 | ~15% | — | — | — | ❌ 过慢终止 |
| 8 | 200ep | ~30% | — | — | — | ❌ 过拟合 |
| 9 | LR=5e-4 | 34.55 | 37.36 | 56.55 | 64.83 | ❌ LR过高 |
| 10 | MLP投影头 | 29.63 | 32.19 | 52.60 | 62.51 | ❌ MLP过拟合 |
| 11 | wd=0.05 | 34.11 | 36.84 | 58.18 | 67.40 | ❌ 正则过强 |
| 12 | 延迟EMA(0.99) | 31.71 | 35.05 | 55.17 | 65.13 | ❌ EMA=0.07% |
| 13 | RandAugment | 24.69 | 25.90 | 46.05 | 56.06 | ❌ 增强过激 |
| 14 | TrivialAugment | 21.82 | 22.98 | 42.29 | 52.37 | ❌ 增强过激 |
| 15 | grad_accum=2 | 29.56 | 31.91 | 52.97 | 62.83 | ❌ 步数减少 |
| 16 | CJ(0.3) | 30.92 | 34.86 | 54.94 | 64.93 | ❌ CJ过强 |
| 17 | w=64 | 32.53 | 35.78 | 55.88 | 64.61 | ❌ 过宽过拟合 |
| 18 | 复现Exp6 | 32.23 | 35.26 | 55.07 | 64.68 | ⚠️ 随机方差 4.42% |
| **19** | **冻结预训练ViT-B/16** | **47.36** | **49.97** | 72.54 | 80.70 | ✅ **突破性改进 +10.71** |
| 20 | 双预训练+全解冻DB | 58.48 | 59.67 | 83.32 | 89.94 | ⚠️ 过拟合 (loss=0.97) |
| **21** | **部分解冻双预训练** | **58.03** | **60.09** | **83.71** | **90.53** | ✅ **最优模型 (loss=0.92)** |

### 推理优化 / Inference Optimization

| 方法 / Method | t2i R@1 | 延迟 / Latency | 加速 / Speedup | 模型大小 / Size | 精度损失 / Accuracy Loss | 评价 |
|---------------|---------|----------------|----------------|-----------------|------------------------|------|
| FP32 基线 | 58.03% | 7.68 ms/sample | 1.0x | 609.7 MB | — | 基准 |
| **BF16** | **58.01%** | **2.01 ms** | **3.82x** | **304.9 MB** | **<0.02pp** | ✅ **最优部署方案** |
| FP16 | 57.96% | 2.10 ms | 3.66x | 304.9 MB | 0.07pp | ✅ 可用 |
| INT8 (仅冻结层) | 42.91% | 68.32 ms | 0.11x | 258.4 MB | -15.12pp | ❌ 精度灾难 |
| INT8 (全量化) | 36.48% | 59.16 ms | 0.13x | 103.0 MB | -21.55pp | ❌ 精度灾难 |
| 层剪枝 ViT-1 | ~0% | — | — | — | ~58pp | ❌ 灾难性失败 |

---

## 🔬 技术细节 / Technical Details

### 模型架构演进 / Model Architecture Evolution

**Exp Baseline (Exp0-18)**：ResNet18 (width=48, ~4.4M params) + 2层 Transformer (4头, d=256, ~2M params) → SigLIP Loss。从头训练，天花板 t2i R@1=36.65%。

**Exp19**：冻结 SigLIP ViT-B/16 (86M params, google/siglip-base-patch16-224) + 可训练 2层 Transformer (d=768, 18.7M params) + image adapter。仅 12 epoch 即超过 Exp6 的 100 epoch 结果。

**Exp20**：ViT 顶层3层解冻 (6M) + **全解冻** DistilBERT (66M) + 投影层。88M 可训练参数导致过拟合 (train_loss 降到 0.0x, val_loss 停滞在 0.94)。

**Exp21 (最优)**：ViT 顶层3层解冻 (6M, LR=1e-5) + DistilBERT **仅顶层2层**解冻 (14M, LR=2e-5) + 投影层 (12M, LR=3e-4)。36M 可训练参数，训练健康无过拟合。

### SigLIP 损失函数 / SigLIP Loss

与 CLIP 使用 softmax 全局归一化不同，SigLIP 对每个图文对独立使用 sigmoid：

$$\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{i,j} \text{softplus}(-s_{ij}(t \cdot z_{ij} + b))$$

- $s_{ij}=+1$（正样本对，对角线），$s_{ij}=-1$（负样本对）
- $t = \exp(\log t)$ 和 $b$ 为可学习参数
- 优势：不依赖 batch size，小 batch 也能提供有效训练信号

### 部分解冻策略 / Partial Unfreeze Strategy

预训练 Transformer 各层形成层次化表示：
- **底层**：捕获通用模式（冻结保护预训练知识不被小数据破坏）
- **顶层**：捕获任务相关语义（微调适应下游任务）
- **差分学习率**：ViT 顶层 LR=1e-5, DistilBERT 顶层 LR=2e-5, 投影层 LR=3e-4

### BF16 推理优化 / BF16 Inference Optimization

- 模型权重 BF16 (`model.to(torch.bfloat16)`)，RTX 4090 Tensor Core 原生支持
- SigLIPLoss 的 `logit_scale.exp()` 保持 FP32 防止溢出
- 预计算全库 embedding (启动时编码 3290 图像+caption)，检索仅需 dot product (<50ms)
- 分离编码器调用：`encode_text()` 和 `encode_image()`，避免运行另一半编码器

---

## 📝 关键结论 / Key Findings

1. **Cosine Warmup** 是从头训练最关键优化 (+8.57% t2i R@1)
2. **预训练视觉编码器** 是突破天花板的关键 (Exp19 vs Exp6: +10.71%)
3. **部分解冻** 优于全解冻 (Exp21 test_loss=0.92 vs Exp20 0.97, 训练更健康)
4. Flickr8k 对数据增强极度敏感 (RandAugment/TrivialAugment/RRC 均严重损害)
5. 更大模型容量在小数据集上反而过拟合 (w=64, MLP proj, 4L+8H+d512)
6. **BF16** 是最实用推理优化 (3.82x 加速, 精度 <0.02pp, 模型减半)
7. 层剪枝/INT8 在无重训练下不可行 (预训练 Transformer 层间依赖强)

---

## 📄 报告与文档 / Reports & Documentation

- **`report/final_report.pdf`** — 正式提交的 LaTeX 最终报告 (包含模型结构、训练配置、实验分析、可视化、推理优化、Demo系统)
- **`experiment_report.html`** — 交互式 HTML 实验报告 (浏览器打开即可浏览 22 组实验详情)
- **`docs/original_assignment/`** — 老师原始作业要求和项目指南

---

## 📚 参考与致谢 / References & Acknowledgments

- [SigLIP](https://arxiv.org/abs/2303.15343) — Zhai et al., "Sigmoid Loss for Language Image Pre-Training", 2023
- [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) — HuggingFace 预训练模型
- [DistilBERT](https://huggingface.co/distilbert-base-uncased) — Sanh et al., "DistilBERT, a distilled version of BERT", 2019
- [Flickr8k](https://illinois.nlplab.org/data/) — Hodosh et al., "Framing Image Description as a Ranking Task"