# 2026-03-19 Night Research: Base Model 选择 & SFT 数据量

---

## 1. 候选 Base Model 对比（≤7B）

### 模型概览

| 模型 | 参数量 | Context | 特点 | 许可证 |
|------|--------|---------|------|--------|
| **Qwen2.5-1.5B** | 1.5B | 32K | 边缘侧最强 1.5B，中英双语 | Apache 2.0 |
| **Qwen2.5-3B** | 3B | 32K | 超越 Phi-3.5-mini 和 MiniCPM3-4B | Apache 2.0 |
| **Qwen2.5-7B** | 7B | 128K | MMLU=74.2, MATH=49.8 | Apache 2.0 |
| **SmolLM3-3B** | 3B | 128K | HuggingFace 最新，outperform LLaMA-3.2-3B 和 Qwen2.5-3B | Apache 2.0 |
| **LLaMA-3.2-1B** | 1.3B | 128K | Meta 出品，轻量 | Llama 3.2 License |
| **LLaMA-3.2-3B** | 3.2B | 128K | Meta 出品 | Llama 3.2 License |
| **Phi-4-mini** | 3.8B | 128K | 微软，推理强，接近 7-9B 水平 | MIT |
| **Gemma-3-1B** | 1B | 32K | Google，极轻量 | Gemma License |
| **Gemma-3-4B** | 4B | 128K | Google，多语言强 | Gemma License |

### 我们的任务特点

```
输入: skill bank (N 行 name+desc) + candidate (完整信息) ≈ 500-3000 tokens
输出: {"operation": "add/merge/discard"} ≈ 10 tokens
任务: 三分类（非生成，非推理）
GRPO: 需要 temperature 采样多个决策
```

**关键需求**:
- JSON 格式遵循（输出严格 JSON）
- 短 context 理解（不需要 128K，32K 足够）
- 三分类准确率（不需要复杂推理）
- LoRA 友好（快速 SFT）
- GRPO 友好（能稳定采样多个不同决策）

### 推荐排名

#### Tier 1: 最佳性价比

**Qwen2.5-3B-Instruct** ⭐ 首选

- 理由:
  - 3B 是三分类任务的甜区——1.5B 可能表达力不足，7B 对三分类过剩
  - Qwen 系列是当前 fine-tuning 生态最成熟的（下载量最高、社区最活跃）
  - 32K context 足够（bank 50 个 skill ≈ 5000 tokens）
  - Apache 2.0 许可证，无限制
  - 中英双语（你的数据有中英混合）
  - GRPO 实战验证（DeepSeek-R1 的蒸馏就是基于 Qwen）
  - Unsloth / LLaMA-Factory / TRL 全部原生支持

**SmolLM3-3B** ⭐ 备选

- 理由:
  - 2026 年 7 月最新发布，benchmark 上 outperform Qwen2.5-3B 和 LLaMA-3.2-3B
  - HuggingFace 官方出品，和 TRL 生态无缝集成
  - 已有 GRPO 训练的实战经验（smol course 有教程）
  - 128K context
- 劣势:
  - 太新，社区 fine-tuning 经验少
  - 中文能力不如 Qwen

#### Tier 2: 可选

**Qwen2.5-1.5B-Instruct** — 如果 GPU 资源紧张或想快速迭代

- 三分类够用，训练最快（~1 小时）
- 但 GRPO 时决策多样性可能不足（模型太小，temperature 采样的变化不够大）

**Phi-4-mini (3.8B)** — 如果看重 JSON 格式遵循

- 微软专门优化了 structured output
- 但 GRPO 生态支持不如 Qwen

**Qwen2.5-7B-Instruct** — 如果要最高准确率

- 7B 对三分类绝对够，但训练时间和显存翻倍
- 适合作为 ablation（证明 3B 够用，7B 提升不大）

#### Tier 3: 不推荐

**LLaMA-3.2-1B/3B** — Qwen 在同参数量上全面超越，且 Llama License 有限制

**Gemma-3-1B/4B** — 中文能力弱，fine-tuning 生态不如 Qwen

---

## 2. SFT 数据量需求

### 行业参考数据

| 来源 | 任务类型 | 模型大小 | 数据量 | 效果 |
|------|---------|---------|--------|------|
| Distillabs 基准测试 | 分类（TREC, Banking77） | 1-3B | 500-1000 | 92%+ accuracy |
| Unsloth 指南 | 指令跟随 | 1.5-7B | 100-500 | 足够 LoRA SFT |
| SuperAnnotate 调研 | 分类/提取 | 3-8B | 100-500 | 足够 |
| Particula 调研 | 简单分类 | 各种 | 100-300/类 | 92% accuracy |
| LLaMA-3.1-8B 实验 | 5 类分类 | 8B | 150 总计 | 92% accuracy（LoRA） |
| LIMA 论文 | 指令跟随 | 65B | 1000 | 接近 GPT-4 |

### 我们的任务分析

```
任务: 三分类 (add / merge / discard)
每类需要: ~100-300 条高质量样本
总计: ~300-1000 条

但我们的 prompt 较长（bank + candidate ≈ 1000-3000 tokens），
模型需要理解 bank 列表和 candidate 的语义关系再做判断，
比单纯的文本分类复杂一些。

推荐: 500-1500 条
最低: 300 条（能跑通但准确率可能不够）
最优: 1000-2000 条（充足但不浪费）
```

### 各模型大小的数据需求

| 模型大小 | 最低数据量 | 推荐数据量 | 训练时间 (1x A800) |
|---------|----------|----------|-------------------|
| 1.5B | 300 条 | 500-1000 条 | ~30 分钟 |
| 3B | 500 条 | 1000-1500 条 | ~1-2 小时 |
| 7B | 800 条 | 1500-2000 条 | ~3-4 小时 |

### 数据质量 vs 数量

**质量远比数量重要。** 关键原则：

1. **经 SkillNet 筛选的 positive 样本**比未筛选的值 3-5 倍
2. **覆盖所有三类决策**，不要 add 占 90%（类别不平衡会让模型只学 add）
3. **覆盖不同 bank 大小**（bank=0, bank=5, bank=20），不同 similarity 区间
4. **dropout 后的变体**也算有效数据（同一条 transition 可以生成多条不同 dropout 的训练样本）

### 推荐的数据构成

```
目标: 1000-1500 条 SFT 样本

来源:
  AutoSkill transitions (positive): ~500-800 条
  × dropout 变体 (每条生成 2 种 dropout): ×2
  = ~1000-1600 条

类别分布（理想）:
  add:     40%  (~400-600 条)
  merge:   40%  (~400-600 条)
  discard: 20%  (~200-300 条)

如果实际分布不平衡（如 add=70%, merge=25%, discard=5%）:
  → 对 merge 和 discard 做 oversampling
  → 或对 add 做 undersampling
```

---

## 3. 推荐方案

### 主方案

```
Base Model: Qwen2.5-3B-Instruct
SFT 数据: 1000-1500 条（经 SkillNet 筛选 + dropout 变体）
训练方法: LoRA (r=16, alpha=32)
训练时间: ~1-2 小时 (1x A800)
GRPO: 在 SFT 基础上继续训
```

### Ablation 方案

| 实验 | 模型 | 数据量 | 目的 |
|------|------|--------|------|
| 主实验 | Qwen2.5-3B | 1000 条 | 最优配置 |
| 模型大小 ablation | Qwen2.5-1.5B | 1000 条 | 证明 3B 比 1.5B 好 |
| 模型大小 ablation | Qwen2.5-7B | 1000 条 | 证明 7B 提升有限 |
| 数据量 ablation | Qwen2.5-3B | 300 条 | 数据量下限 |
| 数据量 ablation | Qwen2.5-3B | 500 条 | 数据量中间 |
| 数据量 ablation | Qwen2.5-3B | 2000 条 | 数据量上限 |
| 替代模型 | SmolLM3-3B | 1000 条 | 最新模型对比 |

---

## 4. LoRA 训练配置参考

```python
from peft import LoraConfig
from trl import SFTTrainer

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=4096,      # bank + candidate + completion
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True,                 # A800 用 bf16 更好
)
```

### 显存估算

| 模型 | Full Fine-tune | LoRA (r=16) | QLoRA (4bit + LoRA) |
|------|---------------|-------------|---------------------|
| 1.5B | ~12 GB | ~6 GB | ~4 GB |
| 3B | ~24 GB | ~10 GB | ~6 GB |
| 7B | ~56 GB | ~18 GB | ~10 GB |

A800 (80GB) 跑 3B LoRA 完全没问题。

---

## Sources

- [12 Small Language Models Benchmark for Fine-tuning (Distillabs)](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
- [SmolLM3-3B (HuggingFace)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [How Much Data for Fine-Tuning (Particula)](https://particula.tech/blog/how-much-data-fine-tune-llm)
- [Fine-tuning with Limited Data Survey](https://arxiv.org/abs/2411.09539)
- [Small Models for Tool Calling](https://arxiv.org/abs/2512.15943)
- [SmolLM3 GRPO Training (HF Course)](https://huggingface.co/learn/smol-course/en/unit1/3)
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
