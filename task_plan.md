# SkillController: Learning to Maintain Hierarchical Skill Banks for LLM Agents

## Project Overview

**Core Idea**: 当前 LLM Agent 的经验/技能管理（如 Training-Free GRPO 的 Controller）依赖 LLM prompt call，缺乏优化保证，导致经验池更新不稳定。我们提出：
1. **Trained Controller** — 用可训练模型替代 LLM prompt call，通过下游 reward 信号学习如何稳定地管理经验池
2. **Hierarchical Skill Bank** — 将扁平经验列表替换为层级结构（Meta-Principles → Domain Strategies → Specific Tactics），支持按需检索、分层更新

**定位**: 这不是 Training-Free GRPO 的改进工作。Training-Free GRPO 是我们采集训练数据的工具之一。我们研究的是一个独立问题：**LLM Agent 如何持续积累和管理可复用的长期技能？**

**参考代码**: `/Users/wentaohu/project/youtu-agent`（Training-Free GRPO 实现）

---

## Phase 1: Problem Validation & Data Collection [CODE COMPLETE — 待运行]

**目标**: 验证 "Controller 不稳定" 确实是问题，并收集训练数据

### Tasks

- [x] 1.1 搭建项目基础结构（config, utils, skill_bank, controller）
- [x] 1.2 集成 youtu-agent rollout pipeline（InstrumentedGRPO wrapper）
- [x] 1.3 实现 experience bank 的稳定性度量指标（StabilityMetrics）
  - rollback_rate: 更新导致 reward 下降的比例
  - skill_churn: ADD+DELETE 操作占比
  - skill_lifecycle: 单条经验的生存周期
  - reward/bank_size trajectory
- [x] 1.5 分析脚本（analyze_stability.py）
  - Operation-reward 相关性（哪种操作导致 reward 下降）
  - Bank 增长分析
  - Experience 波动性分析
- [ ] 1.4 在 math 任务上跑 Training-Free GRPO，记录 (state, action, outcome) 数据
  - state = (experience_bank_t, A_text)
  - action = operations (ADD/DELETE/UPDATE/NONE)
  - outcome = delta_reward (经验池更新前后的 agent 表现变化)

### Decisions

- [x] 决策 1: 用 math 任务（AIME24/DAPO-Math-17k）收集数据
- [x] 决策 2: 稳定性度量 = rollback_rate + churn_rate + skill_lifecycle + reward_trajectory

### 运行环境

- GPU 服务器: `/data/hwt/youtu-agent/`
- Python 环境: `youtu` conda env（已安装 ipython, matplotlib, math-verify）
- LLM API: DeepSeek-chat via `api.qingyuntop.top/v1`（中转站）
- 数据库: SQLite (`test.db`)
- 数据已加载: AIME24, AIME25, DAPO-Math-17k, AFM_web_RL, WebWalkerQA
- Baseline 评估: 运行中（`python scripts/run_eval.py --config_name math/math_AIME24`）（已中途取消）

### How to Run (在 GPU 服务器上)

```bash
# Step 0: 已完成 — 环境配置 + 数据加载
cd /data/hwt/youtu-agent
source activate youtu
python scripts/data/process_training_free_GRPO_data.py

# Step 1: 已完成 — Baseline 评估
python scripts/run_eval.py --config_name math/math_AIME24

# Step 2: 跑 Training-Free GRPO（原版，积累经验）
python scripts/run_training_free_GRPO.py --config_name math_reasoning

# Step 3: 跑 Instrumented 版本（收集 transition 数据）
cd /data/hwt/skillcontroller
python scripts/collect_data.py \
    --config_name math_reasoning \
    --experiment_name math_stability_v1 \
    --save_dir data/collected \
    --held_out_eval \
    --held_out_size 20

# Step 4: 分析稳定性
python scripts/analyze_stability.py \
    --data_path data/collected/math_stability_v1_transitions.json \
    --output_dir data/analysis
```

---

## Phase 2: Hierarchical Skill Bank Design [NOT STARTED]

**目标**: 设计层级技能库的表示、存储和检索机制

### Tasks

- [x] 2.1 定义 Skill Bank 的数据结构（已在 Phase 1 中完成）
  - SkillNode: (id, content, level, parent_id, children_ids, metadata)
  - 层级定义: L0 Meta-Principles, L1 Domain Strategies, L2 Specific Tactics
  - 操作空间: ADD, UPDATE, DELETE, MOVE + snapshot/restore
- [ ] 2.2 设计检索机制
  - 给定 query，检索最相关的 skill 子树/路径
  - 可选方案: embedding-based, LLM-based, 或 hybrid
- [ ] 2.3 设计 skill bank → prompt 的注入策略
  - 全量注入 vs 按需检索注入
  - 层级格式化（缩进、编号）
- [ ] 2.4 Flat → Hierarchical 自动转换
  - 把 Training-Free GRPO 产出的 flat experiences 自动组织成层级结构

### Decisions

- [x] 决策 3: 固定 3 层（META_PRINCIPLE, DOMAIN_STRATEGY, SPECIFIC_TACTIC）
- [x] 决策 4: 严格树结构（非 DAG）
- [ ] 决策 5: 检索方案选择

---

## Phase 3: Trained Controller [NOT STARTED]

**目标**: 设计并训练 Controller 模型

### Tasks

- [x] 3.1 定义 Controller 接口（已在 Phase 1 中完成）
  - Input: ControllerInput(skill_bank_snapshot, candidate_experiences, objectives)
  - Output: ControllerOutput(actions: list[ControllerAction])
  - ControllerAction: (operation, content, target_node_id, level, parent_id, confidence)
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

## Project Structure

```
skillcontroller/                    (独立项目)
├── src/
│   ├── config/base_config.py       SkillControllerConfig 等
│   ├── skill_bank/
│   │   ├── node.py                 SkillNode + SkillLevel (3-level enum)
│   │   └── bank.py                 HierarchicalSkillBank (CRUD/snapshot/serialize)
│   ├── controller/
│   │   └── base.py                 BaseController + ControllerAction + OperationType
│   ├── data_collection/
│   │   ├── instrumented_grpo.py    InstrumentedGRPO (wrapper around youtu-agent)
│   │   ├── recorder.py             TransitionRecorder + TransitionRecord
│   │   └── stability.py            StabilityMetrics
│   └── utils/                      seed, logger
├── scripts/
│   ├── collect_data.py             数据收集入口
│   └── analyze_stability.py        稳定性分析
├── tests/                          test_skill_bank, test_stability
└── pyproject.toml

youtu-agent/                        (上游依赖，不修改)
├── utu/practice/                   Training-Free GRPO 实现
├── utu/eval/                       评估 pipeline
├── configs/                        Hydra 配置
└── scripts/                        运行脚本
```

## Relationship to youtu-agent

- 独立项目，通过 `sys.path.insert(0, youtu_agent_path)` 运行时引用
- 复用: RolloutManager, EvaluationSample, ExperienceCache, ConfigLoader, SimplifiedAsyncOpenAI
- 替换: ExperienceUpdater._group_update/_batch_update → TrainedController
- 替换: flat experience dict → HierarchicalSkillBank
