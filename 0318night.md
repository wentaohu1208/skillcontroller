# 2026-03-18 Night Research Notes

---

## 1. 适合构造 SFT 数据的数据集（按 Domain 分类）

### 原则

AutoSkill 只从"用户给出持久性约束/偏好"的对话中提取 skill。所以需要的数据要么**本身就是多轮有约束的对话**，要么**可以 rewrite 成这种格式**。

### A. 通用对话（直接灌入 AutoSkill）

| 数据集 | 规模 | 特点 | 预计 extraction 成功率 | 获取 |
|--------|------|------|----------------------|------|
| **WildChat-1M** | 1M | 真实 ChatGPT 对话，68 种语言 | ~10-30%（大部分太 generic） | `allenai/WildChat-1M` |
| **LMSYS-Chat-1M** | 1M | 25 个模型的真实对话 | ~10-30% | `lmsys/lmsys-chat-1m` |
| **ShareGPT** | 90K | 用户分享的 ChatGPT 对话 | ~15-30% | HuggingFace |
| **Chatbot Arena** | 33K | 双模型对比 + 用户偏好 | ~20-40%（有偏好信号） | `lmsys/chatbot_arena_conversations` |
| **OpenAssistant** | 161K | 众包多轮对话，有质量标注 | ~20-35% | `OpenAssistant/oasst1` |

### B. 数学 Domain（需要 rewrite）

| 数据集 | 规模 | 内容 | Rewrite 方法 |
|--------|------|------|-------------|
| **GSM8K** | 8.5K | 小学数学应用题 + 推理步骤 | 把解题策略改写成用户约束："以后解应用题都先列方程再验算" |
| **MATH** | 12.5K | 竞赛数学（7 个子类别） | 按类别提取通用策略作为 skill |
| **AIME 24/25** | ~60 | AIME 竞赛题 | 和 Training-Free GRPO 的经验结合 |
| **DAPO-Math-17k** | 17K | 数学 RL 训练数据 | 提取解题模式作为 skill |
| **NuminaMath** | 860K | 竞赛数学 + 解题过程 | 提取通用数学方法论 |
| **MathInstruct** | 262K | 13 个数学数据集混合 | 按 domain 分类提取 |

**Rewrite 示例（数学）**:
```json
{"messages": [
  {"role": "user", "content": "Help me solve competition math problems"},
  {"role": "assistant", "content": "Sure, what kind?"},
  {"role": "user", "content": "From now on, for combinatorics problems: 1) always enumerate small cases first with Python 2) identify the pattern 3) prove the general formula 4) verify with at least 2 independent methods"}
]}
```

### C. 物理 Domain（需要 rewrite）

| 数据集 | 规模 | 内容 | Rewrite 方法 |
|--------|------|------|-------------|
| **ScienceQA** | 21K | K-12 科学多选题（含物理） | 提取物理解题策略 |
| **SciBench** | 695 | 大学物理/化学/数学题 | 提取学科方法论 |
| **GPQA** | 448 | 研究生级别科学题（PhD 难度） | 提取高级推理策略 |
| **TheoremQA** | 800 | 需要定理的科学问题 | 提取"什么时候用什么定理" |

### D. 编程 Domain（需要 rewrite）

| 数据集 | 规模 | 内容 | Rewrite 方法 |
|--------|------|------|-------------|
| **CodeContests** | 13K | 编程竞赛题 | 提取算法策略 |
| **HumanEval/MBPP** | 164/974 | 函数级编程题 | 提取编码规范 |
| **SWE-bench** | 2.3K | 真实 GitHub issue 修复 | 提取 debug/修复策略 |
| **PEP 8 / Google Style Guide** | - | 编码规范文档 | 直接 rewrite 成约束对话 |

### E. 搜索/Web Domain（需要 rewrite）

| 数据集 | 规模 | 内容 | Rewrite 方法 |
|--------|------|------|-------------|
| **WebWalkerQA** | - | Web 导航问答 | 提取搜索策略 |
| **HotpotQA** | 113K | 多跳推理问答 | 提取信息检索和综合策略 |
| **Natural Questions** | 307K | Google 搜索问答 | 提取搜索 query 构造策略 |

### F. 写作 Domain（需要 rewrite）

| 数据集 | 规模 | 内容 | Rewrite 方法 |
|--------|------|------|-------------|
| **NeurIPS/ICML reviewer guidelines** | - | 学术审稿标准 | 改写成写作约束 |
| **Alpaca/Dolly** | 52K/15K | 指令跟随数据 | 筛选含写作约束的 |
| **UltraFeedback** | 64K | 带偏好标注的指令数据 | 提取质量标准 |

### G. LLM 批量合成（最可控，推荐）

```python
domains = ["math", "physics", "coding", "writing", "data_analysis", "design", "teaching", "legal"]
# 每个 domain 用 LLM 生成 50 条"用户给约束"的对话
# → 400 条对话，~95% extraction 成功率
# → 400 条 transition
```

---

## 2. 如何证明 Controller 性能 & 刷什么 Benchmark

### 2.1 直接评估 Controller 的 Metric

| Metric | 含义 | 怎么测 |
|--------|------|--------|
| **Decision Accuracy** | Controller 的决策和"最优决策"的一致率 | 和 oracle（知道 Δreward 的决策）对比 |
| **Rollback Rate** ↓ | 更新 skill bank 后 agent 变差的比例 | 每步 eval，统计 Δperformance < 0 的比例 |
| **Skill Bank Quality** ↑ | 最终 skill bank 辅助 agent 的整体效果 | 用最终 bank 跑 benchmark，比较 accuracy |
| **Convergence Speed** ↑ | 多少步后 skill bank 稳定 | 画 reward vs step 曲线，看何时收敛 |
| **Skill Churn** ↓ | ADD+DELETE 占总操作的比例 | 直接统计 |
| **Pareto Front** | performance vs token usage 的权衡 | 不同 (α,β) 下画 Pareto front |

### 2.2 间接评估：下游 Agent 在 Benchmark 上的表现

Controller 管理的 skill bank 注入 agent 后，刷以下 benchmark：

**数学推理**:
- GSM8K（8.5K 小学数学，accuracy）
- MATH（12.5K 竞赛数学，accuracy）
- AIME 24/25（30 题 AIME，accuracy）

**编程**:
- HumanEval（164 题，pass@1）
- MBPP（974 题，pass@1）
- SWE-bench Verified（394 题，resolve rate）

**通用 Agent**:
- ALFWorld（家务任务，success rate）— SkillRL 用的 benchmark
- WebShop（网购任务，success rate）— SkillRL 用的 benchmark

**Memory/Skill 专用**:
- MemoryAgentBench（ICLR 2026，4 项能力：retrieval, learning, understanding, conflict resolution）
- AMA-Bench（长期 agent memory）
- LoCoMo（300 轮超长对话 memory）

### 2.3 推荐的实验表格设计

```
Table 1: Controller Stability (核心指标)
                      | Rollback Rate ↓ | Churn Rate ↓ | Convergence Step ↓
LLM Controller        |     30%          |    60%       |     N/A
Ours (SFT only)       |     15%          |    35%       |     20
Ours (SFT + GRPO)     |      8%          |    20%       |     12

Table 2: Downstream Agent Performance (间接指标)
Skill Bank Method     | GSM8K | MATH | HumanEval | ALFWorld | WebShop
No skill bank         | xx.x  | xx.x | xx.x      | xx.x     | xx.x
Training-Free GRPO    | xx.x  | xx.x | xx.x      | xx.x     | xx.x
AutoSkill             | xx.x  | xx.x | xx.x      | xx.x     | xx.x
Ours (SFT + GRPO)     | xx.x  | xx.x | xx.x      | xx.x     | xx.x

Table 3: Pareto Front (multi-objective)
                      | accuracy=0.45 时的 token | accuracy=0.40 时的 token
LLM Controller        | 2000                    | 1800
Ours (α=1,β=0.1)     | 1500                    | 1300
Ours (α=1,β=0.3)     | 1100                    | 950
```

---

## 3. RL (GRPO) 的数据从哪来

**GRPO 不需要离线数据集——它是 online 的。** 数据在训练过程中实时生成：

```
每一步 GRPO:
  1. 给 Controller 一个输入 (skill_bank_state, candidate)
  2. Controller 生成 G=5 个决策
  3. 每个决策执行 → eval → 得到 score
  4. 用 scores 计算 advantage → 更新 Controller
```

### 输入从哪来

| 来源 | 方式 |
|------|------|
| **AutoSkill live ingest** | 实时灌对话进 AutoSkill，产出 (state, candidate)，交给 Controller 决策 |
| **缓存的 transitions** | 用之前收集的 transitions，replay state 和 candidate |
| **合成的** | LLM 生成 candidate skill，随机 skill bank state |

### Eval 信号从哪来

| 方式 | 成本 | 质量 |
|------|------|------|
| **LLM-as-Judge** | 每步 ~100 次 API call | 中（有噪声） |
| **Task benchmark** (GSM8K/HumanEval) | 每步跑 20 题 | 高（ground truth） |
| **Skill usage rate** | 零成本（事后统计） | 低（延迟大） |

**推荐**: GRPO 阶段用 LLM-as-Judge（成本可控），最终评估用 task benchmark（ground truth）。

---

## 4. Skill 检索的不确定性如何处理

### 问题

AutoSkill 用 embedding 检索最相关的 skill 注入 agent prompt。检索质量影响 agent 表现，但你只想评估 Controller 的管理能力，不想被检索噪声干扰。

### 解决方案

**方案 A: Oracle Retrieval（推荐用于 ablation）**

绕过检索，直接注入全部 skill bank（和 Training-Free GRPO 一样全量注入）：

```python
# 不检索，直接把所有 skill 塞进 prompt
prompt = agent_instructions + "\n" + "\n".join(all_skills)
```

**优点**: 完全消除检索噪声，Controller 的影响被隔离。
**缺点**: skill bank 大了 prompt 太长，只适合 bank < 30 个 skill 时。

**方案 B: Fixed Retriever（推荐用于主实验）**

所有实验用**同一个固定的检索器**（如 BM25 或固定 embedding model），只变 Controller：

```
实验 1: Fixed Retriever + LLM Controller → baseline
实验 2: Fixed Retriever + Ours (SFT) → 证明 SFT 有效
实验 3: Fixed Retriever + Ours (SFT+GRPO) → 证明 GRPO 有效
```

检索器固定 → 唯一变量是 Controller → 公平对比。

**方案 C: 双层评估**

分别报告：
1. **Skill Bank Quality**（不经过检索，直接测 bank 中 skill 的覆盖率和准确率）
2. **End-to-End Performance**（经过检索，测 agent 最终表现）

```
Table: 分离 Controller 和 Retriever 的贡献
                        | Bank Quality ↑ | Retrieval Precision ↑ | End-to-End Acc ↑
LLM Controller + BM25   |    0.65        |       0.70           |    0.42
Ours + BM25              |    0.78        |       0.70           |    0.48  ← Controller 提升
Ours + Embedding         |    0.78        |       0.85           |    0.53  ← Retriever 也提升
```

**方案 D: 评估时用 Ground Truth Retrieval**

如果测试集有标注"哪个 skill 应该被检索"，直接用标注的 skill，不走检索：

```python
# 测试时
for test_case in benchmark:
    relevant_skill = test_case["ground_truth_skill"]  # 标注好的
    response = agent.run(test_case["query"], skill=relevant_skill)
```

**推荐组合**: 主实验用方案 B（Fixed Retriever），ablation 用方案 A（Oracle）证明 Controller 的独立贡献。

---

## 5. GRPO Training 的详细过程（举例）

### 背景

SFT 训练完成后，Controller 已经学会基本决策（"相似度高→merge，低→add"）。但 SFT 只是模仿 AutoSkill 的决策，可能不是最优的。GRPO 让 Controller 自己探索更好的决策。

### 一步 GRPO 的完整过程

**输入**:
```
当前 Skill Bank (3 skills):
  sk_001: "Python coding standards" (v0.1.2)
  sk_002: "Academic writing" (v0.1.0)
  sk_003: "Data analysis workflow" (v0.1.0)

候选 Skill:
  name: "Code review best practices"
  description: "Enforce 2 reviewers, CI pass, no force push"
  similarity to sk_001: 0.68
  similarity to sk_002: 0.12
  similarity to sk_003: 0.15
```

**Step 1: Controller 生成 G=5 个决策**（temperature=0.7）

```
决策 1: {"operation": "add"}                           ← 新建 sk_004
决策 2: {"operation": "merge", "target": "sk_001"}     ← 合并到 Python 规范
决策 3: {"operation": "discard"}                        ← 丢弃
决策 4: {"operation": "add"}                           ← 新建
决策 5: {"operation": "merge", "target": "sk_001"}     ← 合并到 Python 规范
```

**Step 2: 每个决策执行 → 形成 5 个不同的 Skill Bank**

```
Bank 1 (add):     [sk_001, sk_002, sk_003, sk_004_new]
Bank 2 (merge):   [sk_001_v0.1.3, sk_002, sk_003]
Bank 3 (discard): [sk_001, sk_002, sk_003]  (不变)
Bank 4 (add):     [sk_001, sk_002, sk_003, sk_004_new]
Bank 5 (merge):   [sk_001_v0.1.3, sk_002, sk_003]
```

**Step 3: 各 Bank 跑固定测试集 → 得分**

```
测试集: 10 个固定的任务（5 个编程 + 3 个写作 + 2 个数据分析）

Bank 1 (add):     score = 7.8  (新 skill 对编程任务有帮助)
Bank 2 (merge):   score = 8.2  (合并后 Python skill 更全面，编程分更高)
Bank 3 (discard): score = 7.1  (丢了有用的信息)
Bank 4 (add):     score = 7.5  (和 Bank 1 一样但随机波动)
Bank 5 (merge):   score = 8.0  (和 Bank 2 类似)
```

**Step 4: 计算 Group Relative Advantage**

```
mean_score = (7.8 + 8.2 + 7.1 + 7.5 + 8.0) / 5 = 7.72

advantage_1 (add):     7.8 - 7.72 = +0.08
advantage_2 (merge):   8.2 - 7.72 = +0.48  ← 最高
advantage_3 (discard): 7.1 - 7.72 = -0.62  ← 最低
advantage_4 (add):     7.5 - 7.72 = -0.22
advantage_5 (merge):   8.0 - 7.72 = +0.28
```

**Step 5: Policy Update**

```
merge 的 advantage 最高 (+0.48, +0.28) → 增大 merge 的概率
discard 的 advantage 最低 (-0.62) → 减小 discard 的概率
add 的 advantage 中等 → 微调

更新前 Controller 对这种输入的决策分布: add=40%, merge=35%, discard=25%
更新后 Controller 对这种输入的决策分布: add=30%, merge=55%, discard=15%
```

**经过多步 GRPO，Controller 学会**: "相似度 0.68 的编程相关 skill → merge 比 add 好，discard 最差"

### 和 Training-Free GRPO 的呼应

```
Training-Free GRPO:
  输入: 数学题
  rollout: agent 生成 5 个解法
  reward: 对不对
  学到: 更好的解题经验

我们的 GRPO:
  输入: (skill bank, candidate skill)
  rollout: Controller 生成 5 个管理决策
  reward: 下游 agent 表现
  学到: 更好的管理策略
```

**前者优化"知道什么"，后者优化"怎么管理知道的东西"。结构完全对称。**

---

## 6. Controller Model 选择 & 数据量考虑

### 6.1 1.5B 够不够？

**够，甚至可能过剩。** 原因：

Controller 的任务非常窄——看到 (skill bank, candidate, similarity) → 输出 (add/merge/discard + target)。这本质上是一个**分类+指向问题**，比通用语言理解简单得多。

参考数据点：
- Qwen2.5-1.5B 用 7000 条 GRPO 数据就能在 math reasoning 上提升 20%（Oxen.ai Rust coder 实验）
- DeepSeek-R1-Distill-Qwen-1.5B 用 $42 的算力就能训出强 reasoning 能力
- 1.5B 模型足够理解 "skill bank 有 3 个 skill，候选和 sk_001 相似度 0.74" 这种结构化输入

**也可以更小**：如果用 MLP 方案（17 维特征→决策），几百个参数就够了，连 GPU 都不需要。

### 6.2 需要多少 SFT 数据？

| 模型大小 | SFT 数据需求 | 来源 |
|----------|-------------|------|
| MLP (方案 A) | ~300-500 条 | 几分钟 CPU |
| **1.5B LM (方案 B)** | **~1000-2000 条** | ~2 小时 1xA800 |
| 3B LM | ~2000-5000 条 | ~4 小时 |

**1000-2000 条 SFT 数据对 1.5B 足够**——任务简单（3 类分类 + 指向），不需要学通用语言能力。

### 6.3 GRPO 能激发什么能力？

SFT 后模型只是**模仿 AutoSkill 的决策**。GRPO 能激发的额外能力：

1. **超越教师**: SFT 的上限是 AutoSkill 的决策质量。GRPO 让模型自己探索，可能发现"AutoSkill 在相似度 0.6-0.8 区间倾向 merge，但其实 add 效果更好"
2. **Multi-objective 平衡**: SFT 数据里没有 token 成本的概念。GRPO 的 reward 包含 token penalty，模型学会"该精简时精简"
3. **长期视角**: SFT 每步独立决策。GRPO 的 reward 反映的是更新后的整体 bank 质量，模型可能学会"现在不 add，等后面更好的 candidate"
4. **鲁棒性**: SFT 的决策分布很尖（确定性高）。GRPO 会平滑决策分布，减少极端错误

### 6.4 建议的渐进式方案

```
Phase 1: MLP baseline（几分钟，验证 pipeline）
  → 17 维特征 → 3 类分类
  → 证明 trained controller 比 random 好

Phase 2: 1.5B SFT（2 小时，主实验）
  → AutoSkill 数据 SFT
  → 证明比 LLM controller 更稳定

Phase 3: 1.5B SFT + GRPO（4-8 小时，最终版）
  → GRPO 自我改进
  → 证明超越教师（AutoSkill）

Phase 4 (可选): 0.5B vs 1.5B vs 3B 对比
  → 模型大小 ablation
```

**推荐从 Phase 1 开始**——MLP 几分钟就能跑通整个 pipeline，确认 evaluation、reward 计算都没问题后再上 1.5B。

---

## Sources

- [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
- [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [MATH](https://huggingface.co/datasets/hendrycks/competition_math)
- [MemoryAgentBench (ICLR 2026)](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- [AMA-Bench](https://arxiv.org/abs/2602.22769)
- [SkillRL](https://arxiv.org/abs/2602.08234)
- [GRPO for 1.5B Rust Coder](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo)
- [Agent-R1: End-to-End RL for Agents](https://arxiv.org/abs/2511.14460)
- [Memory Survey](https://arxiv.org/abs/2603.07670)
- [MultiChallenge Benchmark](https://arxiv.org/abs/2501.17399)
