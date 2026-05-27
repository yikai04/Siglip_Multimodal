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

## 📁 项目结构 / Project Structure

```
repository/
├── app_inference.py              # Gradio 推理 Demo（BF16/FP32 自动切换）
├── README.md                     # 本文件
├── requirements.txt              # 依赖列表
├── experiment_report.html        # 交互式实验报告
│
├── siglip/                       # 核心 Python 包
│   ├── __init__.py
│   ├── data/                     # 数据加载（Flickr8kDataset, DistilBERTDataset）
│   ├── loss/                     # SigLIPLoss（sigmoid pairwise loss）
│   ├── utils/                    # SimpleTokenizer, AverageMeter, read_caption_file
│   ├── models/                   # 4 个模型架构
│   ├── inference/                # InferenceEngine（预计算 embedding + 快速检索）
│   └── benchmark/                # 推理优化工具（延迟测量、量化/剪枝）
│
├── scripts/                      # 训练/评估/可视化脚本
│   ├── train_siglip*.py          # 4 个训练脚本
│   ├── eval_exp*.py              # 3 个评估脚本
│   ├── viz_exp*.py               # 3 个可视化脚本
│   ├── benchmark_base.py         # FP32 基线 benchmark
│   ├── quantize_fp16.py          # BF16/FP16 半精度实验
│   ├── quantize_dynamic.py       # INT8 动态量化实验
│   └── prune_layers.py           # 层剪枝实验
│
├── visualization/                # 预训练 SigLIP 可视化 Demo
├── docs/original_assignment/     # 老师原始作业要求
├── report/                       # LaTeX 最终报告
├── Flickr8k/                     # 数据集
├── outputs/                      # 训练 checkpoint + 日志 + benchmark
└── viz_outputs/                  # 可视化输出图片
```

---

## 🔧 安装 / Installation

```bash
git clone https://github.com/yikai04/siglip-flickr8k-retrieval.git
cd repository
pip install -r requirements.txt

# 推理 Demo 需要额外安装 Gradio
pip install gradio>=4.15.0
```

---

## 📊 数据准备 / Data Preparation

下载 [Flickr8k 数据集](https://illinois.nlplab.org/data/) 并放置在 `Flickr8k/` 目录：

```
Flickr8k/
├── Images/           # 8000+ 张 JPG
├── captions.txt      # 全部 caption (CSV 格式)
├── test_captions.txt
├── train_captions.txt
├── val_captions.txt
```

---

## 🚀 使用 / Usage

### 训练 / Training

```bash
# Exp21（最优模型）
python scripts/train_siglip_dual_pretrained_v2.py \
    --data-dir Flickr8k --epochs 50 --batch-size 32 \
    --output-dir outputs/exp21_dual_pretrained_v2
```

### 评估 / Evaluation

```bash
python scripts/eval_exp21.py \
    --checkpoint outputs/exp21_dual_pretrained_v2/best_siglip.pt \
    --data-dir Flickr8k
```

### 推理 Demo / Inference Demo

```bash
python app_inference.py    # http://localhost:7861
# GPU -> BF16 加速，CPU -> FP32 正常推理
```

功能：文本搜图 (Text->Image) + 图片搜文 (Image->Text)，启动时预计算全库 embedding，检索 <50ms。

### 可视化 Demo / Visualization Demo

```bash
python -m visualization.app    # http://localhost:7860
```

---

## 📈 实验结果 / Performance

### 从头训练 (Exp0-18)

| Exp | Change | t2i R@1 | i2t R@1 | Note |
|-----|--------|---------|---------|------|
| 0 | 常数LR基线 | 21.26 | 20.88 | baseline |
| 1 | **Cosine Warmup** | 29.83 | 32.07 | key fix (+8.57) |
| 6 | w=48+CJ+EMA | **36.65** | **39.82** | best scratch |
| 13 | RandAugment | 24.69 | 25.90 | too aggressive |
| 18 | reproduce Exp6 | 32.23 | 35.26 | ~4% variance |

### 预训练模型 (Exp19-21)

| Exp | Vision | Text | Unfreeze | Trainable | t2i R@1 | i2t R@1 | loss |
|-----|--------|------|----------|-----------|---------|---------|------|
| 19 | frozen ViT-B/16 | from-scratch | ViT frozen | 18.7M | 47.36 | 49.97 | 2.48 |
| 20 | ViT top3 | **全解冻** DistilBERT | all DB | 88.2M | 58.48 | 59.67 | 0.97 ⚠️ |
| **21** | ViT top3 | **顶层2层** DB | partial | **36.0M** | **58.03** | **60.09** | **0.92** ✅ |

### 推理优化 / Inference Optimization

| Method | t2i R@1 | Latency | Speedup | Size | Verdict |
|--------|---------|---------|---------|------|---------|
| FP32 | 58.03% | 7.68 ms | 1.0x | 609.7 MB | baseline |
| **BF16** | **58.01%** | **2.01 ms** | **3.82x** | **304.9 MB** | ✅ best |
| FP16 | 57.96% | 2.10 ms | 3.66x | 304.9 MB | ✅ OK |
| INT8 | 42.91% | 68.3 ms | 0.11x | 258.4 MB | ❌ bad |

---

## 🔬 技术细节 / Technical Details

### SigLIP Loss

CLIP 使用 softmax 全局归一化，SigLIP 对每个图文对独立使用 sigmoid：

```
L = (1/B) * sum softplus(-s_ij * (t * z_ij + b))
```

s_ij = +1 (正样本), -1 (负样本), t 和 b 可学习。

### Partial Unfreeze

- **冻结底层**：保护预训练通用知识
- **微调顶层**：仅调整任务相关语义（ViT 3层 + DistilBERT 2层）
- **差分学习率**：顶层 2e-5, 投影层 3e-4

### BF16 Inference

- 模型权重 BF16, SigLIPLoss logit_scale.exp() 保持 FP32
- 预计算全库 embedding, 检索仅需 dot product (<50ms)
- 分离编码器 (`encode_text()` / `encode_image()`)

---

## 📝 关键结论 / Key Findings

1. Cosine Warmup 是从头训练最关键优化 (+8.57%)
2. 预训练视觉编码器是突破天花板的关键 (+10.71%)
3. 部分解冻优于全解冻 (Exp21 > Exp20)
4. Flickr8k 对增强极度敏感
5. BF16 是最实用推理优化 (3.82x, <0.02pp)
6. 层剪枝/INT8 无重训练下不可行

---

## 🙏 致谢 / Acknowledgments

- [SigLIP](https://arxiv.org/abs/2303.15343) — Zhai et al., 2023
- [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [Flickr8k](https://illinois.nlplab.org/data/)