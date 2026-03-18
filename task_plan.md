# SkillController: Learning to Maintain Skill Banks for LLM Agents

## Project Overview

**Core Idea**: 当前 LLM Agent 的 skill/experience 管理完全依赖 LLM prompt call，缺乏优化保证。我们训练一个 Controller 来稳定地管理 skill bank。

**训练路线**: AutoSkill 收集 SFT 数据 → SFT 训练小 LM → GRPO 自我改进

**定位**: Controller 是一个**通用模块**，可插到任何经验管理框架上（AutoSkill、Training-Free GRPO、Reflexion 等）。

---

## 主线 Pipeline

```
Phase 1: 数据收集 ← 当前在做
  WildChat 对话 → AutoSkill → (skill_bank_state, candidate, action) transitions
  + 合成数据 (rewrite math/coding/writing 规范)（待定）
  → training_data_lm.jsonl

Phase 2: SFT 训练
  training_data_lm.jsonl → Fine-tune Qwen-1.5B
  → Controller 学会基本决策模式 (add/merge/discard)

Phase 3: GRPO 自我改进
  SFT 模型生成 G 个决策 → LLM-as-Judge 评分 → Group Relative Advantage 更新
  → Controller 超越"教师"（AutoSkill），学到更优策略

Phase 4: 评估
  → 跨框架 (AutoSkill / Training-Free GRPO / Reflexion)（待定）
  → 跨任务 (math / web search / general chat)
  → Multi-objective Pareto (performance vs token)
```

---

## Phase 1: 数据收集 [进行中]

### 1.1 AutoSkill Pipeline（主力数据源）✅ 代码完成

**数据流**:
```
WildChat (HuggingFace) → prepare_wildchat.py → wildchat.jsonl
  → collect_autoskill_data.py → InstrumentedAutoSkill → transitions.jsonl
  → convert_to_training_data.py → training_data_lm.jsonl
```

**一条 SFT 训练数据**:
```
Input:  "Bank有1个skill: TrueNAS(v0.1.0) | 候选: TrueNAS NAS磁盘拓扑 | 最相似: TrueNAS(score=0.74) | Decide?"
GT:     {"operation": "merge", "target_skill_id": "076a06f2..."}
```

**已验证**: 5 条 transition 成功产出（3 add + 2 merge），格式正确。

**代码位置**: `/Users/wentaohu/project/AutoSkill/skillcontroller_pipeline/`
```
├── instrumented_sdk.py        ✅ InstrumentedAutoSkill wrapper
├── feature_extractor.py       ✅ 17 维 domain-agnostic 特征
├── data_converter.py          ✅ MLP + LM 两种训练数据格式
└── scripts/
    ├── prepare_wildchat.py    ✅ WildChat 下载 + 语言筛选
    ├── collect_autoskill_data.py  ✅ 批量收集 transitions
    └── convert_to_training_data.py  ✅ 格式转换
```

**运行命令**:
```bash
cd /data/hwt/AutoSkill

# Step 1: 准备数据
python -m skillcontroller_pipeline.scripts.prepare_wildchat \
    --num_conversations 5000 --output data/wildchat.jsonl --language English

# Step 2: 收集 transitions（多次 shuffle 增加多样性）
python -m skillcontroller_pipeline.scripts.collect_autoskill_data \
    --input data/wildchat.jsonl \
    --output_dir data/autoskill_transitions \
    --num_runs 3 --shuffle \
    --llm_model deepseek-chat \
    --llm_url https://api.qingyuntop.top/v1 \
    --llm_api_key <key> \
    --embeddings_provider hashing

# Step 3: 转换为训练数据 （还没跑过）
python -m skillcontroller_pipeline.scripts.convert_to_training_data \
    --input_dir data/autoskill_transitions \
    --output_dir data/training_data --format both
```

**预期产出**: 2000 对话 × 3 runs → ~1000-2000 条 transition → ~$15-30 API

### 1.2 合成数据扩充 [TODO]（暂时取消）

WildChat 只有 ~10-30% 对话能触发 skill extraction。用 rewrite 提高成功率：

| 数据源 | 方法 | 预期 |
|--------|------|------|
| Math 经验 (Training-Free GRPO 产出的 G0-G26) | rewrite 成对话格式 | ~27 条 |
| Coding 规范 (PEP8, Google style guide) | rewrite 成用户约束对话 | ~50 条 |
| 写作规范 (NeurIPS guidelines, 学术写作规范) | rewrite 成用户反馈对话 | ~100 条 |
| LLM 合成 (8 domain × 50 条) | LLM 生成"用户给约束"的对话 | ~350 条 |

### 1.3 Training-Free GRPO 数据（补充，有 reward）[暂时取消]

在 Training-Free GRPO 框架下收集的数据有 Δperformance（held-out accuracy），可用于 GRPO 阶段。

**代码位置**: `/Users/wentaohu/project/skillcontroller/src/data_collection/`
**状态**: InstrumentedGRPO 已实现，GPU 服务器上运行中（OOM/API 额度问题需解决）

### 1.4 当前 TODO
- [ ] 调研数据集（极度重要）
- [ ] 充值 API → 跑 2000+ 条 WildChat
- [ ] 实现合成数据 rewrite 脚本（暂时不做）
- [ ] 转换为 SFT 训练格式

---

## Phase 2: SFT 训练 [NOT STARTED]

### 目标

用 AutoSkill 数据训练小 LM，学会基本的 skill bank 管理决策。

### SFT 数据格式

```
Input prompt:
  Current Skill Bank (3 skills):
    [sk_001] python-coding-standards (v0.1.2)
    [sk_002] report-writing-policy (v0.1.0)
    [sk_003] data-analysis-workflow (v0.1.0)

  Candidate: "APA citation formatting: Generate APA 7th edition citations..."
  Most Similar: report-writing-policy (score=0.71)

  Decide: add, merge, or discard?

Completion:
  {"operation": "merge", "target_skill_id": "sk_002"}
```

### 训练配置

```
模型: Qwen2.5-1.5B / LLaMA-3.2-1B
数据: ~1000-2000 条 SFT 样本
方法: LoRA fine-tune
GPU: 1x A800, ~2 小时
Epochs: 1-2
```

### TODO

- [ ] 选择 base model（Qwen-1.5B vs LLaMA-3.2-1B）
- [ ] 实现 SFT 训练脚本（基于 TRL SFTTrainer）
- [ ] 训练 + 评估 SFT 模型

---

## Phase 3: GRPO 自我改进 [NOT STARTED]

### 为什么 GRPO 而非 DPO

- DPO 需要提前配好正负对，受限于已有数据中的最好决策
- GRPO 让模型自己探索新决策，能超越教师（AutoSkill）
- 和 Training-Free GRPO 形成呼应：前者优化经验内容，后者优化管理策略

### GRPO 流程

```
对每个输入 (skill_bank_state, candidate_skill):
  1. Controller 生成 G=5 个决策 (temperature=0.7)
     - 决策 1: add
     - 决策 2: merge sk_001
     - 决策 3: discard
     - ...
  2. 每个决策执行 → LLM-as-Judge eval (固定测试集)
     - 分数: [7.8, 8.2, 7.1, 7.5, 6.9]
  3. advantage_i = score_i - mean(scores)
  4. 增大高 advantage 决策的概率，减小低的
```

### Eval: LLM-as-Judge

```python
def eval_skill_bank(bank, test_conversations):
    """每个决策执行后，用固定测试集 + LLM judge 评分"""
    for conv in test_conversations:
        relevant_skills = retrieve(bank, conv["query"])
        response = llm.generate(conv["query"], skills=relevant_skills)
        score = llm_judge(conv, response)  # 0-10
    return mean(scores)
```

### Multi-Objective Reward

```
reward = α × Δperformance - β × Δtoken_usage_normalized
```

不同 (α, β) 训出不同偏好的 Controller → Pareto front

### 费用估算

```
每步: G=5 决策 × 10 题 × (生成 + judge) = 100 次 API call
100 步: ~10000 calls ≈ $15-30
```

### TODO

- [ ] 实现 LLM-as-Judge eval pipeline
- [ ] 实现 GRPO 训练 pipeline
- [ ] 训练 + 评估 GRPO 模型

---

## Phase 4: 评估 [NOT STARTED]

| 实验 | 目的 |
|------|------|
| 单任务 (math) | trained vs LLM controller 的 accuracy + stability |
| 跨框架 | 同一 controller 插到 AutoSkill / Training-Free GRPO / Reflexion |
| 跨任务迁移 | math 上训 → web search / general chat 上测 |
| 持续学习 | 长期迭代下 skill bank 质量变化 |
| Multi-Objective Pareto | performance vs token usage 的 Pareto front |
| 架构 Ablation | MLP 方案 vs End-to-End LM 方案 |

---

## Controller 架构候选（待决策）

**方案 A: Two-Phase（LLM 特征 + MLP）**
```
候选经验 → LLM(frozen) → 17维数字特征 → MLP(trained) → 决策
```
- 泛化性强，训练数据少（~300 条），可解释
- 信息损失

**方案 B: End-to-End（Fine-tune 小 LM）**
```
skill_bank文本 + 候选经验 → Fine-tuned Qwen-1.5B → 决策JSON
```
- 表达力强，端到端
- 训练数据多（~2000 条），可能 overfit

**当前主线用方案 B（SFT + GRPO 天然适配 LM），方案 A 作为 ablation。**

---

## Related Work Positioning

| 方法 | Skill 表示 | 管理方式 | Trained | Multi-Obj |
|------|-----------|---------|---------|-----------|
| Training-Free GRPO | flat dict | LLM call | No | No |
| AutoSkill | Skill.md | LLM + 启发式 | No | No |
| Voyager | code library | LLM + 验证 | No | No |
| SkillRL | 2-level bank | LLM + 规则 | No (管理) | No |
| MemSkill | memory skills | Trained selector | 部分 | No |
| ASG-SI | skill graph | Verifier + contract | No | No |
| **Ours** | **skill bank** | **SFT + GRPO trained** | **Yes** | **Yes** |

---

## Paper Structure

```
Title: Learning to Maintain Skill Banks for LLM Agents

1. Introduction
2. Related Work
3. Method
   3.1 Skill Bank (flat → 可扩展到 hierarchical)
   3.2 Controller Architecture
   3.3 SFT on AutoSkill Data
   3.4 GRPO Self-Improvement
   3.5 Multi-Objective (Performance vs Token)
4. Experiments
   4.1 Single-task / 4.2 Cross-framework / 4.3 Cross-task
   4.4 Continual / 4.5 Pareto Analysis / 4.6 Ablation
5. Conclusion
```

---

## 环境 & 代码位置

```
GPU 服务器: /data/hwt/
├── AutoSkill/                    AutoSkill + 数据收集 pipeline
│   └── skillcontroller_pipeline/ ← 我们的代码
├── youtu-agent/                  Training-Free GRPO (补充数据源)
└── skillcontroller/              Controller 训练代码 (待实现)

LLM API: DeepSeek-chat via api.qingyuntop.top/v1
Python: autoskill conda env (AutoSkill) / youtu conda env (GRPO)
```

## 关键决策记录

| # | 决策 | 状态 |
|---|------|------|
| 1 | 训练路线: SFT + GRPO（非 DPO） | ✅ 已确定 |
| 2 | 主力数据源: AutoSkill (快、便宜、通用) | ✅ 已确定 |
| 3 | Multi-objective: α × Δperf - β × Δtoken | ✅ 已确定 |
| 4 | Controller 架构: 方案 B (End-to-End LM) 为主，方案 A (MLP) 为 ablation | ✅ 倾向，待最终确认 |
| 5 | Base model: Qwen-1.5B vs LLaMA-3.2-1B | ⬜ 待决策 |
| 6 | Hierarchical skill bank: 何时引入 | ⬜ 待决策（先做 flat） |
