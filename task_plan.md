# SkillController: Learning to Maintain Hierarchical Skill Banks for LLM Agents

## Project Overview

**Core Idea**: 当前 LLM Agent 的经验/技能管理（如 Training-Free GRPO 的 Controller）依赖 LLM prompt call，缺乏优化保证，导致经验池更新不稳定。我们提出：
1. **Trained Controller** — 用可训练模型替代 LLM prompt call，通过下游 reward 信号学习如何稳定地管理经验池
2. **Hierarchical Skill Bank** — 将扁平经验列表替换为层级结构（Meta-Principles → Domain Strategies → Specific Tactics），支持按需检索、分层更新

**定位**: 这不是 Training-Free GRPO 的改进工作。Training-Free GRPO 是我们采集训练数据的工具之一。我们研究的是一个独立问题：**LLM Agent 如何持续积累和管理可复用的长期技能？**

**参考代码**: `/Users/wentaohu/project/youtu-agent`（Training-Free GRPO 实现）

---

## Phase 1: Problem Validation & Data Collection [CODE COMPLETE]

**目标**: 验证 "Controller 不稳定" 确实是问题，并收集训练数据

### Tasks

- [ ] 1.1 搭建项目基础结构（config, utils, db）
- [ ] 1.2 复用 youtu-agent 的 rollout pipeline，跑 Training-Free GRPO baseline
- [ ] 1.3 实现 experience bank 的稳定性度量指标
  - 经验池质量随 step 的变化曲线
  - 单条经验的"生存周期"分析（被 ADD 后多久被 DELETE/UPDATE）
  - Rollback 实验：每步更新后 eval，如果变差则回滚，统计回滚率
- [ ] 1.4 在多个任务上跑 Training-Free GRPO，记录 (state, action, outcome) 数据
  - state = (experience_bank_t, A_text)
  - action = operations (ADD/DELETE/UPDATE/NONE)
  - outcome = delta_reward (经验池更新前后的 agent 表现变化)
- [ ] 1.5 数据分析：哪些类型的操作容易导致 reward 下降？

### Decisions

- [ ] 决策 1: 用哪些任务收集数据？(math, web search, code, ...)
- [ ] 决策 2: 稳定性度量的具体定义

---

## Phase 2: Hierarchical Skill Bank Design [NOT STARTED]

**目标**: 设计层级技能库的表示、存储和检索机制

### Tasks

- [ ] 2.1 定义 Skill Bank 的数据结构
  - SkillNode: (id, content, level, parent_id, children_ids, metadata)
  - 层级定义: L0 Meta-Principles, L1 Domain Strategies, L2 Specific Tactics
  - 操作空间: ADD(content, parent_id, level), UPDATE(node_id, content), DELETE(node_id), MOVE(node_id, new_parent_id)
- [ ] 2.2 设计检索机制
  - 给定 query，检索最相关的 skill 子树/路径
  - 可选方案: embedding-based, LLM-based, 或 hybrid
- [ ] 2.3 设计 skill bank → prompt 的注入策略
  - 全量注入 vs 按需检索注入
  - 层级格式化（缩进、编号）
- [ ] 2.4 实现 Hierarchical Skill Bank 模块

### Decisions

- [ ] 决策 3: 固定层数(3) vs 动态生长
- [ ] 决策 4: 严格树结构 vs DAG（一条 skill 多个父节点）
- [ ] 决策 5: 检索方案选择

---

## Phase 3: Trained Controller [NOT STARTED]

**目标**: 设计并训练 Controller 模型

### Tasks

- [ ] 3.1 定义 Controller 的输入/输出形式化
  - Input: (skill_bank_state, candidate_experiences)
  - Output: (operation_type, target_node, content, level)
- [ ] 3.2 Controller 架构设计
  - 方案 A: 小型 LM fine-tune（如 Qwen-1.5B）
  - 方案 B: LLM feature extraction + 分类头
  - 方案 C: Encoder-based model（对 skill bank 和 A_text 做 encoding）
- [ ] 3.3 训练方案设计
  - 路线 A: 离线监督学习（Phase 1 收集的数据，正样本=delta_reward>0）
  - 路线 B: RL 训练（Controller 作为 agent, reward=delta_performance）
  - 路线 C: DPO/偏好学习（好的更新 vs 差的更新配对）
- [ ] 3.4 实现训练 pipeline
- [ ] 3.5 实现推理 pipeline（trained controller 替换 LLM controller）

### Decisions

- [ ] 决策 6: Controller 架构选择
- [ ] 决策 7: 训练方案选择
- [ ] 决策 8: Controller 的泛化策略（跨任务迁移怎么做）

---

## Phase 4: Experiments & Evaluation [NOT STARTED]

**目标**: 全面评估系统

### Tasks

- [ ] 4.1 单任务实验
  - Baseline: flat + LLM controller (= Training-Free GRPO)
  - Ours: hierarchical + trained controller
  - Ablation: flat + trained, hierarchical + LLM
- [ ] 4.2 跨任务迁移实验
  - 在 math 上训的 controller 用到 web search / code 上
  - 证明 controller 学到的是 "如何管理知识" 而非 domain knowledge
- [ ] 4.3 持续学习实验
  - 长期迭代（10+ epochs）下 skill bank 的质量变化
  - 对比 trained vs LLM controller 的稳定性
- [ ] 4.4 Skill Bank 分析
  - 可视化层级结构
  - 经验的聚类/分布分析
- [ ] 4.5 与其他方法对比
  - Voyager-style skill library
  - RAG-based retrieval
  - Reflexion

### Decisions

- [ ] 决策 9: 评估指标定义
- [ ] 决策 10: 计算预算规划

---

## Phase 5: Paper Writing [NOT STARTED]

### Proposed Structure

```
Title: Learning to Maintain Hierarchical Skill Banks for LLM Agents

1. Introduction
2. Related Work (Skill Libraries, Meta-learning, In-context Learning)
3. Method
   3.1 Hierarchical Skill Bank
   3.2 Trained Controller
   3.3 Training Pipeline
4. Experiments
   4.1 Single-task / 4.2 Cross-task / 4.3 Continual / 4.4 Analysis
5. Conclusion
```

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │   Trained Controller_θ   │
                    │                         │
                    │  Input:                  │
                    │  - Skill Bank (tree)     │
                    │  - Candidate A_text      │
                    │                         │
                    │  Output:                 │
                    │  - Operations on tree    │
                    │  - (op, node, content,   │
                    │     level)               │
                    │                         │
                    │  Training Signal:        │
                    │  - Downstream Δreward    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Hierarchical Skill Bank │
                    │                         │
                    │  L0: Meta-Principles     │
                    │  L1: Domain Strategies   │
                    │  L2: Specific Tactics    │
                    └────────────┬────────────┘
                                 │ retrieve & inject
                    ┌────────────▼────────────┐
                    │     Frozen Policy LLM    │
                    │      (Agent Rollout)     │
                    └────────────┬────────────┘
                                 │ reward signal
                    ┌────────────▼────────────┐
                    │    Reward / Verifier     │
                    └─────────────────────────┘
```

## Relationship to youtu-agent

复用 youtu-agent 的组件:
- `RolloutManager` — rollout 执行
- `BaseBenchmark` / `BaseProcesser` — 评估 pipeline
- `EvaluationSample` — 数据模型
- `SimplifiedAsyncOpenAI` — LLM 调用
- `ExperienceCache` — 缓存机制
- Verification functions (math, web)

新增/替换的组件:
- `HierarchicalSkillBank` — 替换 flat experience dict
- `TrainedController` — 替换 LLM-based ExperienceUpdater
- `SkillRetriever` — 新增，从 skill bank 检索相关技能
- `ControllerTrainer` — 新增，Controller 训练 pipeline
- `StabilityMetrics` — 新增，稳定性度量
