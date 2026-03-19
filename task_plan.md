# SkillController: Learning to Maintain Skill Banks for LLM Agents

## Project Overview

**Core Idea**: 当前 LLM Agent 的 skill/experience 管理完全依赖 LLM prompt call，缺乏优化保证。我们训练一个 Controller 来稳定地管理 skill bank。

**训练路线**: AutoSkill 收集数据 → SkillNet 质量评估筛选 → SFT 训练小 LM → GRPO 自我改进

**定位**: Controller 是一个**通用模块**，可插到任何经验管理框架上（AutoSkill、Training-Free GRPO、Reflexion 等）。

---

## 主线 Pipeline

```
Phase 1: 数据收集 + 质量筛选 ← 当前在做
  WildChat 对话 → AutoSkill 提取 skill → SkillNet evaluate 质量门控
  → 高质量 transitions → training_data_lm.jsonl

Phase 2: SFT 训练
  training_data_lm.jsonl → Fine-tune Qwen-1.5B
  → Controller 学会基本决策模式 (add/merge/discard)

Phase 3: GRPO 自我改进
  SFT 模型生成 G 个决策 → eval 评分 → Group Relative Advantage 更新
  → Controller 超越"教师"（AutoSkill），学到更优策略

Phase 4: 评估
  → SkillsBench / ALFWorld / WebShop（下游 agent 效果）
  → 稳定性指标（rollback rate, churn rate）
  → Multi-objective Pareto (performance vs token)
```

---

## Phase 1: 数据收集 + SkillNet 质量筛选 [进行中]

### 1.1 AutoSkill 数据收集 Pipeline ✅ 代码完成

**数据流**:
```
WildChat → prepare_wildchat.py → wildchat.jsonl
  → collect_autoskill_data.py → InstrumentedAutoSkill → transitions.jsonl
  → convert_to_training_data.py → training_data_lm.jsonl
```

**已验证**: 5 条 transition 成功产出（3 add + 2 merge），格式正确。

**代码位置**: `/Users/wentaohu/project/AutoSkill/skillcontroller_pipeline/`

**运行命令**:
```bash
cd /data/hwt/AutoSkill

python -m skillcontroller_pipeline.scripts.prepare_wildchat \
    --num_conversations 5000 --output data/wildchat.jsonl --language English

python -m skillcontroller_pipeline.scripts.collect_autoskill_data \
    --input data/wildchat.jsonl \
    --output_dir data/autoskill_transitions \
    --num_runs 3 --shuffle \
    --llm_model deepseek-chat \
    --llm_url https://api.qingyuntop.top/v1 \
    --llm_api_key <key> \
    --embeddings_provider hashing

python -m skillcontroller_pipeline.scripts.convert_to_training_data \
    --input_dir data/autoskill_transitions \
    --output_dir data/training_data --format both
```

### 1.2 SkillNet 质量评估筛选 SFT 数据 [✅ 代码完成 — 方式A]

**目的**: AutoSkill 的决策（add/merge/discard）不全是好决策。用 SkillNet `evaluate()` 评估 candidate skill 质量，结合 AutoSkill 的决策和 similarity score，**筛选出好决策作为 SFT 正样本，坏决策作为负样本或丢弃**。

**核心逻辑**: Skill 质量 + AutoSkill 决策 → 判断决策好坏，similarity作为参数可以调整

| Skill 质量 | AutoSkill 决策 | similarity | 判断 | 原因 |
|-----------|--------------|-----------|------|------|
| Poor | add / merge | - | ❌ 坏决策 | 加了垃圾 skill |
| Poor | discard | - | ✅ 好决策 | 丢了垃圾 skill |
| Good | add | - | ✅ 好决策 | 加了好 skill |
| Good | merge | - | ✅ 好决策 | 合并好 skill |
| Good | discard | > 0.7 | ✅ 好决策 | 好 skill 但已有类似的，去重正确 |
| Good | discard | ≤ 0.7 | ❌ 坏决策 | 好 skill 没重复却被丢了 |

**SkillNet evaluate() 五维度**:

| 维度 | 问的是什么 | Good | Poor |
|------|-----------|------|------|
| **Safety** | 用了会不会出事 | 无危险操作 | 有 rm -rf |
| **Completeness** | 写全了没有 | 步骤完整 | 只有一句话 |
| **Executability** | 能不能照着做 | 指令清晰 | 模糊建议 |
| **Maintainability** | 好不好改/组合 | 模块化 | 耦合严重 |
| **Cost-awareness** | 费不费钱 | 不需要 API | 每次调 10 次 GPT-4 |

每维度三档：**Good / Average / Poor**。Poor ≥ 2 个 → 判定为低质量 skill。（这里评判高低质量的方法有待考量，可以尝试一个多层次，带有优先级的考量）

**两种评估方式**:

---

#### 方式 A: 后置评估（推荐）

先让 AutoSkill **正常跑完所有 transition**（不拦截），跑完后用 SkillNet **批量打标签**。

```
Step 1: AutoSkill 正常跑 → 500 条 transition（不干预）
Step 2: 对每条 transition 用 SkillNet evaluate + similarity → 打 positive/negative 标签
Step 3: 保留 positive 的作为 SFT 数据
```

```python
# 后置评估脚本（新建 scripts/label_transitions.py）
for t in transitions:
    quality = skillnet_evaluate(t["candidate"])
    action = t["action"]
    similarity = t["similar_hits"][0]["score"] if t["similar_hits"] else 0

    poor_count = count_poor(quality)

    if poor_count >= 2 and action in ["add", "merge"]:
        t["label"] = "negative"     # 加了垃圾 → 坏决策
    elif poor_count >= 2 and action == "discard":
        t["label"] = "positive"     # 丢了垃圾 → 好决策
    elif poor_count < 2 and action in ["add", "merge"]:
        t["label"] = "positive"     # 加了好 skill → 好决策
    elif poor_count < 2 and action == "discard" and similarity > 0.7:
        t["label"] = "positive"     # 好 skill 但已有类似 → 去重正确
    elif poor_count < 2 and action == "discard" and similarity <= 0.7:
        t["label"] = "negative"     # 好 skill 没重复却丢了 → 坏决策

sft_data = [t for t in transitions if t["label"] == "positive"]
```

**优势**:
- 不干预 AutoSkill 的行为，transition 完全自然
- 能同时拿到正样本和负样本（负样本可用于 DPO 或分析）
- 实现简单，不改数据收集代码，只加一个后处理脚本
- 可以反复调整筛选标准（不用重跑数据收集）

**劣势**:
- skill bank 可能被垃圾 skill 污染，后续 transition 的 state 不干净
- 负样本中 AutoSkill 基于被污染的 state 做的决策，可能不是真正的"坏决策"

---

#### 方式 B: 交错评估（拦截）

在每条 transition 时**实时评估**，低质量 skill 被拦截不加入 bank，保持 bank 干净。

```
Step 0: candidate → evaluate → Poor → 拦截，不 add → bank 保持干净
Step 1: candidate → evaluate → Good → 允许 → 正常执行 AutoSkill 决策
Step 2: 基于干净的 bank 继续…
```

```python
# 在 InstrumentedAutoSkill._instrumented_upsert() 中加门控
def _instrumented_upsert(self, cand, *, user_id, metadata):
    # 1. SkillNet 评估
    eval_result = self.evaluator.evaluate(export_as_skill(cand))
    poor_count = count_poor(eval_result)

    if poor_count >= 2:
        # 拦截：不执行原始决策，强制 discard
        self.records.append({..., "action": "discard_by_filter", "eval": eval_result})
        return None  # 不加入 bank

    # 2. 质量通过，正常执行原始决策
    result = self.original_upsert(cand, user_id=user_id, metadata=metadata)
    self.records[-1]["skill_quality"] = eval_result
    return result
```

**优势**:
- skill bank 始终干净，后续 transition 的 state 不被污染
- 产出的 transition 质量更高（都是基于干净 state 的决策）

**劣势**:
- 干预了 AutoSkill 的行为，transition 不完全自然（混入了你的门控决策）
- 拿不到负样本（低质量 skill 被拦截了，看不到 AutoSkill 对它的原始决策）
- 需要改 instrumented_sdk.py

---

#### 对比总结

| | 方式 A: 后置评估 | 方式 B: 交错评估 |
|---|---|---|
| **Bank 状态** | 可能被污染 | 始终干净 |
| **Transition 自然性** | ✅ 完全自然 | ⚠️ 被门控干预 |
| **正样本** | ✅ 有 | ✅ 有 |
| **负样本** | ✅ 有 | ❌ 没有 |
| **实现复杂度** | 低（后处理脚本） | 中（改 instrumented_sdk） |
| **可重复性** | ✅ 高（可反复调筛选标准） | ⚠️ 低（改标准需重跑） |
| **适合** | SFT + 后续 DPO | 纯 SFT |

**当前选择：方式 A（后置评估）。** 方式 B 作为后续可选。

**方式 A 已实现的代码**:

```
/Users/wentaohu/project/AutoSkill/skillcontroller_pipeline/
├── skill_quality_gate.py          ✅ SkillNet evaluate 封装 + 标签逻辑
│   - evaluate_candidate(): 导出 SKILL.md → SkillNet 五维度评估
│   - label_transition(): skill 质量 + 决策 + similarity → positive/negative
│   - label_all_transitions(): 批量标注所有 transitions
└── scripts/
    └── label_transitions.py       ✅ 后置标注脚本
```

**运行命令（在 AutoSkill 数据收集完成后）**:

```bash
cd /data/hwt/AutoSkill

# pip install skillnet-ai（如果还没装）
python -m skillcontroller_pipeline.scripts.label_transitions \
    --input_dir data/autoskill_transitions \
    --output_dir data/labeled_transitions \
    --api_key <key> \
    --base_url https://api.qingyuntop.top/v1 \
    --model deepseek-chat \
    --similarity_threshold 0.7
```

**产出**:

```
data/labeled_transitions/
├── all_labeled.jsonl      ← 所有 transition（带 label 和 quality 字段）
├── sft_positive.jsonl     ← 好决策（用于 SFT 训练）
└── sft_negative.jsonl     ← 坏决策（用于分析或 DPO）
```

---

**SkillNet evaluate() 技术细节**:

```python
from skillnet_ai import SkillEvaluator, EvaluatorConfig, Skill

config = EvaluatorConfig(
    api_key="<your-key>",
    base_url="https://api.qingyuntop.top/v1",
    model="deepseek-chat",
    run_scripts=False,
    cache_dir="./eval_cache"
)
evaluator = SkillEvaluator(config)

skill, err = Skill.from_path("/path/to/skill_dir")
result = evaluator.evaluate(skill)
# → {"safety": {"level": "Good", "reason": "..."}, "completeness": {...}, ...}
```

**成本估算**: 每条 transition 1 次 LLM call (~$0.005)，2000 条 ≈ $10。

### 1.3 SkillNet analyze() 关系分析 [TODO — 可选]

**目的**: 在整批数据收集完后，分析最终 skill bank 里 skill 之间的关系，回溯标注哪些决策对/错。

```python
from skillnet_ai import SkillNetClient
client = SkillNetClient(api_key="<key>")

# 分析最终 skill bank 中所有 skill 的关系
relationships = client.analyze(skills_dir="./final_skill_bank")
# → [{"source": "python-standards", "target": "code-review", "type": "similar_to", "reason": "..."}]
```

**四种关系类型 → 对应正确决策**:

| SkillNet 关系 | 含义 | 正确决策 | 错误决策 |
|--------------|------|---------|---------|
| `similar_to` | 功能等价 | **merge** | add（创建了重复） |
| `compose_with` | 互补，可组合使用 | **add** | discard（丢了有用的） |
| `depend_on` | A 依赖 B | **add** | discard |
| `belong_to` | A 是 B 的子组件 | **merge** 到 B | add（该归属不该独立） |

**用法**: 收集完所有 transition 后跑一次 analyze()，回溯标注：
```
最终 bank 里有 similar_to 的两个 skill
  → 回溯找到是哪一步 add 创建了重复
  → 标记那条 transition 为负样本
```

### 1.4 SkillNet search() 验证 [TODO — 可选]

**目的**: 检查 AutoSkill 提取的 skill 在 SkillNet 的 150K+ 库里有没有类似的（社区验证过的）。

```python
hits = client.search(q="python coding standards type hints", mode="vector", threshold=0.8)
# 如果 SkillNet 库里有高度相似的 → 说明这个 skill 是有价值的
```

### 1.5 合成数据扩充 [TODO — 暂时取消]

WildChat 只有 ~10-30% 对话能触发 skill extraction。可用 rewrite 提高成功率。

### 1.6 当前 TODO

- [ ] 充值 API → 跑 2000+ 条 WildChat
- [ ] 集成 SkillNet evaluate() 到 InstrumentedAutoSkill（质量门控）
- [ ] 转换为 SFT 训练格式
- [ ] (可选) 收集完后跑 SkillNet analyze() 标注正负样本

---

## Phase 2: SFT 训练 [NOT STARTED]

### 目标

用经过 SkillNet 质量筛选的 AutoSkill 数据训练小 LM。

### SFT 数据格式

```
Input prompt:
  Current Skill Bank (3 skills):
    [sk_001] python-coding-standards (v0.1.2) [Quality: Good]
    [sk_002] report-writing-policy (v0.1.0) [Quality: Good]
    [sk_003] data-analysis-workflow (v0.1.0) [Quality: Average]

  Candidate: "APA citation formatting: Generate APA 7th edition citations..."
  Candidate Quality: Good (safety=Good, completeness=Good, executability=Average, ...)
  Most Similar: report-writing-policy (score=0.71)

  Decide: add, merge, or discard?

Completion:
  {"operation": "merge", "target_skill_id": "sk_002"}
```

### 训练配置

```
模型: Qwen2.5-1.5B / LLaMA-3.2-1B
数据: ~1000-2000 条 SFT 样本（经 SkillNet 筛选）
方法: LoRA fine-tune
GPU: 1x A800, ~2 小时
Epochs: 1-2
```

### TODO

- [ ] 选择 base model
- [ ] 实现 SFT 训练脚本（基于 TRL SFTTrainer）
- [ ] 训练 + 评估 SFT 模型

---

## Phase 3: GRPO 自我改进 [NOT STARTED]

### GRPO 流程

```
对每个输入 (skill_bank_state, candidate_skill):
  1. Controller 生成 G=5 个决策 (temperature=0.7)
  2. 每个决策执行 → eval 评分
  3. advantage_i = score_i - mean(scores)
  4. 增大高 advantage 决策的概率，减小低的
```

### Eval 信号来源（按优先级）

| 方式 | 信号质量 | 成本 |
|------|---------|------|
| SkillsBench pytest | 最高（ground truth） | 高（需跑 agent + Docker） |
| SkillNet evaluate() | 中（skill 质量，非决策质量） | 低（1 次 LLM call） |
| LLM-as-Judge | 中 | 中（每步 ~100 次 call） |

### Multi-Objective Reward

```
reward = α × Δperformance - β × Δtoken_usage_normalized
```

### TODO

- [ ] 实现 eval pipeline（SkillNet evaluate / LLM-as-Judge / SkillsBench）
- [ ] 实现 GRPO 训练 pipeline
- [ ] 训练 + 评估 GRPO 模型

---

## Phase 4: 评估 [NOT STARTED]

### Benchmark

| Benchmark | 测什么 | 怎么测 |
|-----------|--------|--------|
| **SkillsBench** | agent 用 skill 完成任务的能力 | 把我们的 skill bank 导出为 SKILL.md → 放进 task → pytest 验证 |
| **ALFWorld** | 家务 agent（SkillNet 已验证） | 用 skill bank 辅助 agent |
| **WebShop** | 电商 agent（SkillNet 已验证） | 用 skill bank 辅助 agent |
| **GSM8K / MATH** | 数学推理 | skill bank 注入 prompt |
| **HumanEval** | 编程 | skill bank 注入 prompt |
| **MemoryAgentBench** | memory 管理能力（ICLR 2026） | 4 项能力测评 |

### 稳定性指标

| Metric | 含义 |
|--------|------|
| Rollback Rate ↓ | 更新后 agent 变差的比例 |
| Churn Rate ↓ | ADD+DELETE 占总操作的比例 |
| Convergence Step ↓ | 多少步后 skill bank 稳定 |
| SkillNet Quality Score ↑ | bank 中所有 skill 的平均质量分 |

---

## Controller 架构候选（待决策）

**方案 A: Two-Phase（LLM 特征 + MLP）**
```
候选经验 → LLM(frozen) → 17维数字特征 → MLP(trained) → 决策
```

**方案 B: End-to-End（Fine-tune 小 LM）**
```
skill_bank文本 + 候选经验 + 质量评估 → Fine-tuned Qwen-1.5B → 决策JSON
```

**当前主线用方案 B（SFT + GRPO 天然适配 LM），方案 A 作为 ablation。**

---

## Related Work Positioning

| 方法 | Skill 表示 | 管理方式 | Trained | 质量评估 |
|------|-----------|---------|---------|---------|
| Training-Free GRPO | flat dict | LLM call | No | No |
| AutoSkill | Skill.md | LLM + 启发式 | No | No |
| SkillNet | Skill Package | 五维度评估 + 关系图 | No | **Yes** |
| SkillRL | 2-level bank | LLM + 规则 | No | No |
| **Ours** | **skill bank** | **SFT + GRPO** | **Yes** | **SkillNet 门控** |

---

## Paper Structure

```
Title: Learning to Maintain Skill Banks for LLM Agents

1. Introduction
2. Related Work
3. Method
   3.1 Skill Bank
   3.2 SkillNet Quality Gate (五维度评估门控)
   3.3 Controller Architecture (End-to-End LM)
   3.4 SFT on AutoSkill Data
   3.5 GRPO Self-Improvement
   3.6 Multi-Objective (Performance vs Token)
4. Experiments
   4.1 SkillsBench / ALFWorld / WebShop
   4.2 Stability Analysis
   4.3 Cross-framework / Cross-task
   4.4 Pareto Analysis / Ablation
5. Conclusion
```

---

## 环境 & 代码位置

```
GPU 服务器: /data/hwt/
├── AutoSkill/                    AutoSkill + 数据收集 pipeline
│   └── skillcontroller_pipeline/ ← 数据收集代码
├── SkillNet/                     SkillNet 质量评估
│   └── skillnet-ai/             ← evaluate(), analyze(), search()
├── youtu-agent/                  Training-Free GRPO (补充数据源)
└── skillcontroller/              Controller 训练代码 (待实现)

LLM API: DeepSeek-chat via api.qingyuntop.top/v1
Python: autoskill conda env / youtu conda env
```

## 关键决策记录

| # | 决策 | 状态 |
|---|------|------|
| 1 | 训练路线: SFT + GRPO（非 DPO） | ✅ 已确定 |
| 2 | 主力数据源: AutoSkill (快、便宜、通用) | ✅ 已确定 |
| 3 | 质量门控: SkillNet evaluate() 交错评估 | ✅ 已确定 |
| 4 | Multi-objective: α × Δperf - β × Δtoken | ✅ 已确定 |
| 5 | Controller 架构: 方案 B (End-to-End LM) 为主 | ✅ 倾向 |
| 6 | Base model: Qwen-1.5B vs LLaMA-3.2-1B | ⬜ 待决策 |
| 7 | 评估 benchmark: SkillsBench + ALFWorld + WebShop | ⬜ 待确认 |
