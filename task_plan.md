# SkillController: Learning to Maintain Hierarchical Skill Banks for LLM Agents

## Project Overview

**Core Idea**: 当前 LLM Agent 的经验/技能管理（如 Training-Free GRPO 的 Controller）依赖 LLM prompt call，缺乏优化保证，导致经验池更新不稳定。我们提出：
1. **Trained Controller** — 两阶段架构：LLM 提取结构化特征 + 小模型做决策，通过下游 reward 信号学习管理策略
2. **Hierarchical Skill Bank** — 将扁平经验列表替换为层级结构（Meta-Principles → Domain Strategies → Specific Tactics）

**定位**: 这不是 Training-Free GRPO 的改进工作。Controller 是一个**通用模块**，可插到任何经验管理框架（Training-Free GRPO、AutoSkill、Reflexion 等）上。我们研究的是独立问题：**LLM Agent 如何持续积累和管理可复用的长期技能？**

**参考代码**: `/Users/wentaohu/project/youtu-agent`（Training-Free GRPO 实现）

---

## Core Method: Two-Phase Controller

### 设计原理

Training-Free GRPO 的断裂点：LLM 既负责理解语义，又负责做管理决策，但管理决策没有优化信号。我们的方案把这两件事拆开：

```
候选经验(文本) ──→ Phase A (LLM) ──→ 结构化特征(数字) ──→ Phase B (小模型) ──→ 操作决策
                    不训练                                    训练
                    擅长语义理解                              擅长从数据中学决策规律
```

- **Phase A: Feature Extractor (LLM, frozen)** — 把文本语义压缩成 domain-agnostic 的数字特征
- **Phase B: Decision Model (trained, 小模型/MLP)** — 根据特征做操作决策，从 Δreward 信号中学习

### Phase A 特征定义

| 特征 | 来源 | 含义 |
|------|------|------|
| `top1_similarity` | LLM/embedding | 候选经验和 graph 中最相似节点的相似度 |
| `top2_similarity` | LLM/embedding | 第二相似节点的相似度 |
| `top1_node_level` | graph 查询 | 最相似节点在哪一层 |
| `top1_relation` | LLM 判断 | same / refinement / complementary / unrelated |
| `candidate_abstractness` | LLM 打分 0-1 | 0=具体技巧, 1=抽象原则 |
| `graph_size` | 代码统计 | 当前 graph 节点数 |
| `level_counts` | 代码统计 | L0/L1/L2 各多少节点 |
| `recent_rewards` | recorder | 最近几步的 Δreward |
| `reward_trend` | 线性回归 | reward 在涨还是在跌 |
| `steps_since_last_add` | 代码统计 | 多久没 ADD 了 |
| `steps_since_last_delete` | 代码统计 | 多久没 DELETE 了 |

**这些特征是 domain-agnostic 的**——在 math/web search/code 上含义相同。

### Phase B Decision Model

输入: 特征向量（~15 维）
输出: (operation, target_node, level)

### 训练数据格式

```jsonl
{"features": {"top1_similarity": 0.65, "graph_size": 10, ...}, "action": {"operation": "ADD", "target_node": "S0", "level": 1}, "outcome": {"delta_reward": 0.05, "label": "positive"}}
{"features": {"top1_similarity": 0.82, "graph_size": 12, ...}, "action": {"operation": "UPDATE", "target_node": "S3", "level": 1}, "outcome": {"delta_reward": -0.03, "label": "negative"}}
```

### 训练方式选项

- **二分类**: 输入 features → 预测这个 action 是 positive/negative
- **偏好学习**: 同一 graph state 下，正样本 action 优于负样本 action
- **回归**: 直接预测 delta_reward

推理时：枚举所有可能的 action，选 score 最高的。

---

## 冷启动 & 训练数据收集流程（基于 Training-Free GRPO）

### 完整流程

```
Phase 1: 冷启动 — 获得初始 skill graph
  Training-Free GRPO (原版, LLM Controller)
    → flat experiences: {G0: "...", G1: "...", ..., G26: "..."}
    → LLM 一次性分层 → 初始 skill graph（tree 结构）

Phase 2: 数据收集 — 获得训练数据
  Instrumented GRPO (带初始 skill graph)
    → 每步: LLM Controller 做决策（同时记录 features + action + Δreward）
    → 输出: training_data.jsonl

Phase 3: 训练
  training_data.jsonl → 训 Decision Model

Phase 4: 部署 & 验证
  用 trained Decision Model 替换 LLM Controller → 验证稳定性和 reward
```

### Phase 1 详解: 冷启动

**Step 1**: 跑 Training-Free GRPO 原版

```bash
python scripts/run_training_free_GRPO.py --config_name math_reasoning
```

产出: `configs/agents/practice/math_practice_agent.yaml`（含 flat experiences G0-G26）

**Step 2**: LLM 一次性把 flat → hierarchical

```
输入: 27 条 flat experiences
LLM prompt: "请将这些经验组织成 L0(抽象原则) → L1(领域策略) → L2(具体技巧) 三层结构"
输出: 初始 skill graph
```

示例结果：
```
S0 "多方法验证" (L0)
├── S3 "组合: 枚举小案例 + 公式交叉验证" (L1)
│   ├── S8 "Burnside: 检查 cycle 长度" (L2)
│   └── S9 "容斥: 用补集法验证" (L2)
├── S4 "几何: 坐标法 + 综合法双重验证" (L1)
│   └── S10 "对称性: 先建对称坐标系" (L2)
S1 "计算验证" (L0)
├── S5 "Python 暴力验证" (L1)
└── S6 "代码块自包含" (L1)
S2 "问题分解" (L0)
└── S7 "多约束提前合并" (L1)
```

### Phase 2 详解: 数据收集

在初始 skill graph 上跑 Instrumented GRPO（更多 epochs，收集足够数据）：

```bash
python scripts/collect_data.py \
    --config_name math_reasoning \
    --experiment_name math_data_collection \
    --epochs 3 --rollout_data_truncate 500 --batch_size 25 \
    --held_out_eval --held_out_size 20
```

每个 step 记录一条 TransitionRecord，然后转换为训练数据：

```
TransitionRecord (每 step 一条)
  → 拆成每条 operation 一个样本
  → 对每条: LLM 提特征 + 取 action + 取 Δreward
  → 输出一条训练样本到 training_data.jsonl
```

**注意**: Δreward 是整个 step 的整体效果，不是单条 operation 的。这是 credit assignment 的近似。

### 数据量估算

```
3 epochs × (500/25) batches = 60 steps
每 step ~10 条 operation → 600 条训练样本
其中约一半正/一半负 → 300 对可用于偏好学习
API 费用: ~$60-100
```

---

## Phase 1: Problem Validation & Data Collection [CODE COMPLETE — 待运行]

**目标**: 验证 "Controller 不稳定" 确实是问题，并收集训练数据

### Tasks

- [x] 1.1 搭建项目基础结构（config, utils, skill_bank, controller）
- [x] 1.2 集成 youtu-agent rollout pipeline（InstrumentedGRPO wrapper）
- [x] 1.3 实现 experience bank 的稳定性度量指标（StabilityMetrics）
- [x] 1.5 分析脚本（analyze_stability.py）
- [ ] 1.4 在 math 任务上跑 Training-Free GRPO，记录 transition 数据

### Decisions

- [x] 决策 1: 用 math 任务（AIME24/DAPO-Math-17k）收集数据
- [x] 决策 2: 稳定性度量 = rollback_rate + churn_rate + skill_lifecycle + reward_trajectory

### 运行环境

- GPU 服务器: `/data/hwt/youtu-agent/`
- Python 环境: `youtu` conda env（已安装 ipython, matplotlib, math-verify）
- LLM API: DeepSeek-chat via `api.qingyuntop.top/v1`（中转站）
- 数据库: SQLite (`test.db`)
- 数据已加载: AIME24, AIME25, DAPO-Math-17k, AFM_web_RL, WebWalkerQA

### How to Run (在 GPU 服务器上)

```bash
# Step 0: 已完成 — 环境配置 + 数据加载

# Step 1: 跑 Training-Free GRPO 原版（冷启动，获得 flat experiences）
cd /data/hwt/youtu-agent
python scripts/run_training_free_GRPO.py --config_name math_reasoning
# 产出: configs/agents/practice/math_practice_agent.yaml

# Step 2: LLM 把 flat experiences → 初始 skill graph（一次性）
# TODO: 实现 scripts/build_initial_graph.py

# Step 3: 跑 Instrumented 版本（收集 transition 数据）
cd /data/hwt/skillcontroller
python scripts/collect_data.py \
    --config_name math_reasoning \
    --experiment_name math_stability_v1 \
    --save_dir data/collected \
    --held_out_eval --held_out_size 20

# Step 4: 分析稳定性
python scripts/analyze_stability.py \
    --data_path data/collected/math_stability_v1_transitions.json \
    --output_dir data/analysis

# Step 5: 转换 transition → 训练数据
# TODO: 实现 scripts/build_training_data.py
# TransitionRecord → LLM 提特征 → training_data.jsonl
```

---

## Phase 2: Hierarchical Skill Bank Design [PARTIALLY DONE]

**目标**: 设计层级技能库的表示、存储和检索机制

### Tasks

- [x] 2.1 定义 Skill Bank 的数据结构（SkillNode, HierarchicalSkillBank）
- [ ] 2.2 设计检索机制
- [ ] 2.3 设计 skill bank → prompt 的注入策略
- [ ] 2.4 Flat → Hierarchical 自动转换（LLM 一次性分层）

### Decisions

- [x] 决策 3: 固定 3 层（META_PRINCIPLE, DOMAIN_STRATEGY, SPECIFIC_TACTIC）
- [x] 决策 4: 严格树结构（非 DAG）
- [ ] 决策 5: 检索方案选择

---

## Phase 3: Trained Controller [NOT STARTED]

**目标**: 实现两阶段 Controller 并训练

### Tasks

- [x] 3.1 定义 Controller 接口（BaseController, ControllerAction）
- [ ] 3.2 实现 Feature Extractor（Phase A）
  - LLM 提取语义特征（similarity, relation, abstractness）
  - 代码计算统计特征（graph_size, level_counts, reward_trend）
- [ ] 3.3 实现 Decision Model（Phase B）
  - 模型架构: MLP / 小 transformer
  - 输入: ~15 维特征向量
  - 输出: (operation, target_node, level)
- [ ] 3.4 实现训练数据构造 pipeline
  - TransitionRecord → 拆 operations → LLM 提特征 → training_data.jsonl
- [ ] 3.5 实现训练 pipeline
  - 方案: 二分类 / 偏好学习 / 回归（都实现，实验选最佳）
- [ ] 3.6 实现推理 pipeline（trained controller 替换 LLM controller）

### Decisions

- [x] 决策 6: 两阶段架构（LLM Feature Extractor + Trained Decision Model）
- [ ] 决策 7: Decision Model 具体架构（MLP vs 小 transformer）
- [ ] 决策 8: 训练方式（二分类 vs 偏好学习 vs 回归）

---

## Phase 4: Experiments & Evaluation [NOT STARTED]

**目标**: 全面评估系统

### Tasks

- [ ] 4.1 单任务实验（math）
  - Baseline: flat + LLM controller (= Training-Free GRPO)
  - Ours: hierarchical + trained controller
  - Ablation: flat + trained, hierarchical + LLM
- [ ] 4.2 跨框架实验
  - 同一个 trained controller 插到不同框架上
  - Training-Free GRPO / AutoSkill / Reflexion
  - 证明 controller 是通用模块
- [ ] 4.3 跨任务迁移实验
  - math 上训的 controller 用到 web search 上
  - 证明学到的是管理策略而非 domain knowledge
- [ ] 4.4 持续学习实验
  - 长期迭代下 skill bank 的质量变化
  - 对比 trained vs LLM controller 的稳定性
- [ ] 4.5 Skill Bank 分析
  - 可视化层级结构
  - 经验的聚类/分布分析

### Decisions

- [ ] 决策 9: 评估指标定义
- [ ] 决策 10: 计算预算规划

---

## Phase 5: Paper Writing [NOT STARTED]

### Proposed Structure

```
Title: Learning to Maintain Hierarchical Skill Banks for LLM Agents

1. Introduction
2. Related Work
   - Skill Libraries (Voyager, SkillRL, SAGE, AutoSkill, ASG-SI)
   - Memory Management (MemSkill, MemGPT)
   - Training-Free GRPO, Reflexion, ExpeL
3. Method
   3.1 Hierarchical Skill Bank
   3.2 Two-Phase Controller (Feature Extractor + Decision Model)
   3.3 Cold Start & Training Pipeline
4. Experiments
   4.1 Single-task / 4.2 Cross-framework / 4.3 Cross-task / 4.4 Continual / 4.5 Analysis
5. Conclusion
```

### Related Work Positioning

| 方法 | Skill 表示 | 管理方式 | 层级 | Trained |
|------|-----------|---------|------|---------|
| Training-Free GRPO | flat dict | LLM call | No | No |
| AutoSkill | Skill.md | LLM + 启发式 | No | No |
| Voyager | code library | LLM + 执行验证 | No | No |
| SkillRL | 2-level bank | LLM + 规则 | Yes | No (管理) |
| MemSkill | memory skills | Trained selector + LLM designer | No | 部分 (选择) |
| ASG-SI | skill graph | Verifier + contract | Yes | No |
| **Ours** | **3-level tree** | **LLM features + trained model** | **Yes** | **Yes (管理)** |

---

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │     Two-Phase Controller     │
                    │                             │
                    │  Phase A: LLM (frozen)       │
                    │  候选经验 + graph → 特征向量   │
                    │                             │
                    │  Phase B: Decision Model (θ) │
                    │  特征向量 → (op, node, level) │
                    │  训练信号: Δreward            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Hierarchical Skill Bank    │
                    │                             │
                    │   L0: Meta-Principles        │
                    │   L1: Domain Strategies      │
                    │   L2: Specific Tactics       │
                    └──────────────┬──────────────┘
                                   │ retrieve & inject
                    ┌──────────────▼──────────────┐
                    │      Frozen Policy LLM       │
                    │       (Agent Rollout)        │
                    └──────────────┬──────────────┘
                                   │ reward signal
                    ┌──────────────▼──────────────┐
                    │      Reward / Verifier       │
                    └─────────────────────────────┘
```

## Project Structure

```
skillcontroller/
├── src/
│   ├── config/base_config.py
│   ├── skill_bank/
│   │   ├── node.py                 SkillNode + SkillLevel
│   │   └── bank.py                 HierarchicalSkillBank
│   ├── controller/
│   │   ├── base.py                 BaseController + ControllerAction
│   │   ├── feature_extractor.py    Phase A: LLM → 特征 (TODO)
│   │   └── decision_model.py       Phase B: 特征 → 操作 (TODO)
│   ├── data_collection/
│   │   ├── instrumented_grpo.py    InstrumentedGRPO wrapper
│   │   ├── recorder.py             TransitionRecorder
│   │   └── stability.py            StabilityMetrics
│   └── utils/
├── scripts/
│   ├── collect_data.py             数据收集
│   ├── analyze_stability.py        稳定性分析
│   ├── build_initial_graph.py      flat → hierarchical (TODO)
│   ├── build_training_data.py      transition → 训练数据 (TODO)
│   └── train_controller.py         训练 Decision Model (TODO)
├── tests/
└── pyproject.toml
```
