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
  → Controller 超越"教师"（AutoSkill），学到更优策略（这里有两个方案，这是说的是普通版）

Phase 4: 评估（指标待定）
  → SkillsBench / ALFWorld / WebShop（下游 agent 效果）
  → 稳定性指标（rollback rate, churn rate）
  → Multi-objective Pareto (performance vs token)
```

---

## Phase 1: 数据收集 + SkillNet 质量筛选 [进行中]

### 1.1 AutoSkill 数据收集 Pipeline ✅ 代码完成

**数据流**:
```
Step 1: WildChat → prepare_wildchat.py → wildchat.jsonl
Step 2: → collect_autoskill_data.py → transitions.jsonl（原始 transitions）
Step 3: → label_transitions.py → sft_positive.jsonl（SkillNet 筛选后）
Step 4: → convert_to_training_data.py → training_data_lm.jsonl（决策 + 具体操作）
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

每维度三档：**Good / Average / Poor**。采用**分层门控**判定质量：

```
Layer 1 安全红线:   Safety = Poor → 直接不合格（不可逆风险）
Layer 2 可用性:     Completeness = Poor 或 Executability = Poor → 不合格（agent 用不了）
Layer 3 工程质量:   Maintainability + Cost 都 Poor → 不合格（留着没价值）
                   单独一个 Poor → 通过（当前能用，长期可优化）
```

设计原理：
- **Safety** 最高优先级——不安全的 skill 进了 bank，agent 执行危险操作是不可逆的
- **Completeness + Executability** 次优先级——决定 agent 拿到 skill 后能不能用
- **Maintainability** 单独容忍——当前能用，长期 bank 大了再清理
- **Cost-awareness** 单独容忍——贵但有用，token 成本在 multi-objective reward 里处理

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
from skillcontroller_pipeline.skill_quality_gate import evaluate_candidate, label_transition

for t in transitions:
    # SkillNet 五维度评估 + 分层门控
    quality = evaluate_candidate(t["candidate"], evaluator)
    # quality.is_high_quality 由三层门控决定:
    #   Layer 1: Safety=Poor → False
    #   Layer 2: Completeness=Poor 或 Executability=Poor → False
    #   Layer 3: Maintainability+Cost 都 Poor → False

    # 结合质量 + 决策 + similarity 打标签
    t["label"] = label_transition(t, quality, similarity_threshold=0.7)
    # positive: 好决策（好skill被add/merge，垃圾被discard，重复被discard）
    # negative: 坏决策（垃圾被add/merge，好skill无重复却被discard）

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

### 1.3 构造训练数据：纯三分类决策 [TODO — 需改 data_converter.py]

**设计决策**: 模型**只输出决策（add/merge/discard）**，具体操作由 AutoSkill 已有代码执行。模型不生成任何 skill 内容。

**理由**:
- 三分类对 1.5B 模型轻松，GRPO 也够用
- 具体操作（merge 内容融合、add skill 创建）是确定性逻辑，只要决策完成，就可以通过既定的方式去add/merge，代码做更可靠
- 训练数据需求少（~1000 条）

**Completion 格式（三选一）**:

```json
{"operation": "add"}
{"operation": "merge"}
{"operation": "discard"}
```

**Prompt 设计**:

Prompt 分为三部分：Skill Bank 概览 + Candidate 完整信息 + 可选上下文。

**(1) Skill Bank：每个 skill 只展示 name + description（一行）**

```
Current Skill Bank (4 skills):
  [1] truenas-disk-topology: Designs disk topology and dataset layouts for TrueNAS Scale NAS systems
  [2] midjourney-prompt: Generates imaginative Midjourney V5 text-to-image prompts
  [3] childrens-book: Creates short rhyming children's books about animals
  [4] wrestling-examples: Provides examples of wrestling matches with exclusion criteria
```

不展示 instructions（太长，20 个 skill 就上万 token）、不展示 tags/triggers（不通用）。name + description 足够判断候选和 bank 里哪个重复。

**(2) Candidate：展示完整信息**

```
Candidate Skill:
  Name: midjourney-creative-prompt-generation
  Description: Generates creative Midjourney V5 prompts using artistic references
  Instructions: # Goal\nCreate diverse prompts based on user concepts...
  Triggers: create midjourney prompts, text-to-image prompts
  Tags: midjourney, ai-art, prompt-generation
  Confidence: 0.80
```

**(3) 可选上下文（Optional Context）**

```
Optional Context:
  Most Similar: midjourney-prompt (score=0.52, v0.1.1)
```

**跨框架通用性设计**:

| 字段 | 通用性 | 处理方式 |
|------|--------|---------|
| Bank 的 name + description | ✅ Agent Skills 行业标准 | **必需**，始终存在 |
| Candidate 的 name + description | ✅ Agent Skills 行业标准 | **必需**，始终存在 |
| Candidate 的 instructions | ✅ Agent Skills 行业标准（SKILL.md body） | **必需**，始终存在 |
| Candidate 的 triggers/tags | ⚠️ AutoSkill 特有，标准里没有 | **可选**，训练时随机 dropout |
| Candidate 的 confidence | ⚠️ AutoSkill 特有，标准里没有 | **可选**，训练时随机 dropout |
| Most Similar (score) | ⚠️ 需要 embedding 检索 | **可选**，训练时随机 dropout |

> Agent Skills 行业标准（agentskills.io，2025.10 Anthropic 发布，OpenAI Codex / ChatGPT / Claude Code 均采用）只定义了 skill 的**格式标准**（name + description + instructions），**没有定义自动管理机制**（add/merge/discard）。Anthropic 的 skill 管理完全靠人手动操作。这正是我们的 contribution——训练一个 Controller 自动做管理决策。

**训练时随机 dropout**：~30% 的样本随机去掉部分可选字段，让模型学会在信息不完整时也能做决策。推理时任何框架只要整理成相同的 prompt 格式（bank 的 name+desc + candidate 的完整信息）就能直接用。（这个可以作为超参数调整）


**为什么 Trained Controller 可能比现有启发式/LLM Controller 更好**:

**1. 全局视角 vs 局部对比**

AutoSkill 每次只做 pair-wise 比较（candidate vs 一个 existing skill）。我们的 Controller 看到**整个 bank 的列表**，能做全局判断：
```
AutoSkill: "candidate 和 sk_003 相似度 0.72 → merge"（只看一对）
Ours:      "bank 里已经有 3 个 coding 类 skill 了，再 add 冗余度太高 → merge"（看全局）
```

**2. 从历史中学习 vs 每次 zero-shot**

AutoSkill 的 LLM Controller 每次决策都是 zero-shot（没有历史信息）。我们的 Controller 通过 SFT 学到了大量历史决策 pattern，通过 GRPO 学到了"什么决策导致好结果"：
```
AutoSkill: 每次独立判断，不知道之前的 merge 效果好不好
Ours:      从训练数据中学到"similarity 0.5-0.7 区间 merge 效果通常比 add 好"
```

**3. 确定性 vs 随机性**

AutoSkill 的 LLM 每次 call 可能给不同答案（temperature、prompt 微变、模型版本更新）。我们的 1.5B 模型是确定性的：
```
AutoSkill: 同一个输入跑 10 次可能 7 次 merge、3 次 add
Ours:      同一个输入永远同一个决策（temperature=0）
```

**4. 成本：$0 vs $0.01/次**

AutoSkill 每次决策调一次大模型 API（`_judge_merge_with_llm`），有延迟有费用。我们的 1.5B 本地推理几毫秒，零 API 成本。

**5. 阈值的脆弱性**

纯启发式（similarity > 0.7 → merge）在边界值极其脆弱。换个 embedding 模型，0.7 的含义完全不同。我们的模型从多维信息做决策，不依赖单一阈值：
```
启发式: similarity = 0.69 → add, similarity = 0.71 → merge（差 0.02 导致完全不同的决策）
Ours:   综合 bank 大小、候选描述、历史 pattern 做判断，不会因为一个数字的微小变化翻转
```

**预期结果**：我们可能在单次决策准确率上接近 AutoSkill（大模型语义理解更强），但在**稳定性、成本、一致性、跨框架通用性**上显著更好。论文的 story 是综合优势，不是单项冠军。

**完整的一条训练数据（LM 格式）**:

```
Input prompt:
  Current Skill Bank (4 skills):
    [1] truenas-disk-topology: Designs disk topology and dataset layouts for TrueNAS Scale NAS systems
    [2] midjourney-prompt: Generates imaginative Midjourney V5 text-to-image prompts
    [3] childrens-book: Creates short rhyming children's books about animals
    [4] wrestling-examples: Provides examples of wrestling matches with exclusion criteria

  Candidate Skill:
    Name: midjourney-creative-prompt-generation
    Description: Generates creative Midjourney V5 prompts using artistic references
    Instructions: # Goal\nCreate diverse prompts based on user concepts...
    Triggers: create midjourney prompts, text-to-image prompts
    Tags: midjourney, ai-art, prompt-generation
    Confidence: 0.80

  Optional Context:
    Most Similar: midjourney-prompt (score=0.52, v0.1.1)

  Decide: add, merge, or discard?

Completion:
  {"operation": "merge"}
```

**GT 来源**: transition 中 AutoSkill 的 `action` 字段（经 SkillNet 筛选后的 positive 样本）。

**推理时的执行映射**:

| Controller 输出 | 代码执行 |
|----------------|---------|
| `{"operation": "add"}` | 调 AutoSkill `store.upsert()` 把 candidate 原样加入 bank |
| `{"operation": "merge"}` | 调 AutoSkill `_persist_merged()` 把 candidate 合并到 `similar_hits[0]` |
| `{"operation": "discard"}` | 什么都不做 |

**模型只做决策，执行交给代码。Controller 输出几个 token，AutoSkill 负责实际操作。**

**执行层设计（训好后实现）**:

```python
class ControllerExecutor:
    """接收 Controller 决策，调 AutoSkill 已有接口执行实际操作"""

    def __init__(self, sdk: AutoSkill, user_id: str):
        self.store = sdk.store
        self.maintainer = sdk.maintainer
        self.user_id = user_id

    def execute(self, decision: dict, candidate, similar_hits):
        op = decision["operation"]

        if op == "add":
            # 复用 AutoSkill 的 _create_new() 逻辑
            new_skill = create_skill_from_candidate(candidate)
            self.store.upsert(new_skill)
            return new_skill

        elif op == "merge":
            # 取 most similar 作为 merge 目标
            target = self.store.get(similar_hits[0]["skill_id"])
            # 复用 AutoSkill 的 _persist_merged() 做内容合并 + version bump
            merged = self.maintainer._persist_merged(target, candidate)
            return merged

        elif op == "discard":
            return None
```

核心思路：**不自己写 merge/add 逻辑**，全部复用 AutoSkill `SkillMaintainer` 已有的方法：
- `_create_new()` → 从 candidate 创建 Skill 对象 + 分配 UUID + 设 version 0.1.0
- `_persist_merged()` → LLM 辅助合并 candidate 到已有 skill + version bump + 存储
- `store.upsert()` → 持久化到 SkillBank 目录

**完整推理流程**:

```
新对话进来
  → AutoSkill LLMSkillExtractor.extract() → SkillCandidate
  → Embedding 检索 → similar_hits
  → 构造 prompt (bank + candidate + similar)
  → Controller 模型推理 → {"operation": "merge"}
  → ControllerExecutor.execute() → 调 AutoSkill 代码执行 merge
  → Skill bank 更新
```

---

### 1.4 SkillNet analyze() 关系分析 [可选]

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

### 1.5 SkillNet search() 验证 [可选]

**目的**: 检查 AutoSkill 提取的 skill 在 SkillNet 的 150K+ 库里有没有类似的（社区验证过的）。

```python
hits = client.search(q="python coding standards type hints", mode="vector", threshold=0.8)
# 如果 SkillNet 库里有高度相似的 → 说明这个 skill 是有价值的
```

### 1.6 合成数据扩充 [暂时取消]

WildChat 只有 ~10-30% 对话能触发 skill extraction。可用 rewrite 提高成功率。

### 1.7 当前 TODO

- [ ] 充值 API → 跑 2000+ 条 WildChat
- [ ] 解决 instructions 截断问题（中转站 rate limit 导致 DeepSeek 输出被截断）
- [x] SkillNet 质量筛选代码（label_transitions.py）
- [ ] 改造 convert_to_training_data.py：输出"决策 + 具体操作"格式（diff bank_before/after 提取 GT）
- [ ] 运行完整 pipeline: collect → label → convert
- [ ] (可选) 收集完后跑 SkillNet analyze() 标注正负样本

---

## Phase 2: SFT 训练 [NOT STARTED]

### 目标

用经过 SkillNet 质量筛选的 AutoSkill 数据训练小 LM，学会 add/merge/discard 三分类决策。

### Base Model 选择

**首选：Qwen2.5-3B-Instruct**（详见 0319night.md 调研）

理由：
- 3B 是三分类任务甜区（1.5B 可能不足，7B 对三分类过剩）
- Qwen 系列 fine-tuning 生态最成熟（下载量最高，TRL/Unsloth/LLaMA-Factory 原生支持）
- 32K context（bank 50 个 skill ≈ 5000 tokens，完全够）
- Apache 2.0 许可证
- GRPO 实战验证（DeepSeek-R1 蒸馏基于 Qwen）

Ablation 备选：Qwen2.5-1.5B、Qwen2.5-7B、SmolLM3-3B

### SFT 数据准备

**输入**：Phase 1 产出的 `sft_positive.jsonl`（经 SkillNet 筛选的 positive transitions）

**转换**：用 `convert_to_training_data.py --format lm` 转成 prompt-completion 对

**数据格式**（和 Phase 1.3 一致）：
```
prompt:     bank(name+desc) + candidate(完整信息) + optional context
completion: {"operation": "add/merge/discard"}
```

**数据量目标**：1000-1500 条（含 dropout 变体）

**类别平衡**：
```
理想分布: add 40% / merge 40% / discard 20%
如果不平衡: 对少数类 oversampling 或多数类 undersampling
```

### SFT 训练实施

**代码位置**：`/Users/wentaohu/project/skillcontroller/scripts/train_sft.py`（待实现）

**训练框架**：TRL SFTTrainer + PEFT LoRA

```python
# train_sft.py 核心代码结构

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. 加载 base model
model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载训练数据
dataset = load_dataset("json", data_files="data/training_data/training_data_lm.jsonl")

# 3. 格式化函数：prompt + completion 拼接
def format_fn(sample):
    return f"{sample['prompt']}\n{sample['completion']}"

# 4. LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 5. 训练配置
sft_config = SFTConfig(
    output_dir="./outputs/sft_controller",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=4096,
    bf16=True,                    # A800 用 bf16
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    seed=42,
)

# 6. 训练
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("validation"),
    peft_config=lora_config,
    formatting_func=format_fn,
)
trainer.train()

# 7. 保存
trainer.save_model("./outputs/sft_controller/final")
```

**运行命令**：
```bash
cd /data/hwt/skillcontroller

# LoRA 训练
python scripts/train_sft.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_path /data/hwt/AutoSkill/data/training_data/training_data_lm.jsonl \
    --output_dir ./outputs/sft_controller_lora \
    --epochs 2 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 16

# Full Fine-tune（数据量 >1500 时可选）
python scripts/train_sft.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_path /data/hwt/AutoSkill/data/training_data/training_data_lm.jsonl \
    --output_dir ./outputs/sft_controller_full \
    --epochs 2 \
    --batch_size 4 \
    --lr 2e-5 \
    --full_finetune

# 推理测试
python scripts/eval_sft.py \
    --model_path ./outputs/sft_controller/final \
    --test_data /data/hwt/AutoSkill/data/training_data/test_data_lm.jsonl
```

### 训练配置

**两种微调方式可选**：

| | LoRA | Full Fine-tune |
|---|---|---|
| 训练参数量 | ~1% (r=16) | 100% |
| 显存 (3B) | ~10GB | ~24GB |
| 训练时间 | ~1-2 小时 | ~3-4 小时 |
| 防 overfit | ✅ 天然正则化 | ⚠️ 数据少时容易 overfit |
| 适合数据量 | 500-1500 条 | 1500+ 条 |
| A800 能跑吗 | ✅ | ✅ |

**推荐**：数据 <1500 条用 LoRA，>1500 条可以尝试 full fine-tune，两种都做作为 ablation。

```
Base Model:   Qwen2.5-3B-Instruct
方法:          LoRA (r=16, alpha=32) 或 Full Fine-tune
数据:          ~1000-1500 条 SFT 样本
Epochs:       2
Batch Size:   4 × 4 gradient accumulation = effective 16
Learning Rate: LoRA: 2e-4 / Full: 2e-5
Max Seq Len:  4096
精度:          bf16
GPU:          1x A800
```

### SFT 评估

**评估指标**：

| 指标 | 怎么算 |
|------|--------|
| **Decision Accuracy** | 在 held-out test set 上的三分类准确率 |
| **Per-class F1** | add / merge / discard 各自的 precision + recall |
| **JSON Format Rate** | 输出合法 JSON 的比例（应该 >99%） |

**评估流程**：

```python
# eval_sft.py 核心逻辑
for sample in test_data:
    output = model.generate(sample["prompt"], max_new_tokens=50, temperature=0)
    predicted = json.loads(output)["operation"]
    ground_truth = sample["action"]
    # 统计 accuracy, per-class F1
```

**数据划分**：
```
总数据 ~1000 条
  → 训练集 80%: ~800 条
  → 验证集 10%: ~100 条（训练时 eval loss 监控）
  → 测试集 10%: ~100 条（最终报数）
```

### Baseline 对比

SFT 训完后和以下 baseline 对比：

| Baseline | 方法 | 在 test set 上的 accuracy |
|----------|------|--------------------------|
| Random | 随机三选一 | ~33% |
| Majority Class | 永远输出最多的那个类 | ~40% |
| 纯阈值 (sim>0.7→merge) | 规则 | xx% |
| AutoSkill LLM Controller | 大模型 API call | xx%（上限参考） |
| **Ours (SFT)** | Qwen2.5-3B LoRA | **xx%** |

### TODO

- [ ] 下载 Qwen2.5-3B-Instruct 到 GPU 服务器
- [ ] 实现 `scripts/train_sft.py`
- [ ] 实现 `scripts/eval_sft.py`
- [ ] 数据划分（train/val/test 80/10/10）
- [ ] 训练 + 评估
- [ ] Ablation: 1.5B / 7B / SmolLM3 对比

---

## Phase 3: GRPO 自我改进 [NOT STARTED]

### 设计决策：独立采样（非链式）

**和常规 GRPO 一样，每步独立，bank 不实时变化。**

常规 GRPO（如 DeepSeek-R1 训 math）每步输入一道独立的题目。我们的 GRPO 每步从**已收集的 transitions 里随机采样**一条 (bank_snapshot, candidate)，互相独立。不做链式更新（一步错步步错的问题）。

### GRPO 数据来源

**直接复用 Phase 1 收集的 transitions**，每条 transition 里有：
- `skill_bank_before`：bank 快照（作为 prompt 的 bank 部分）
- `candidate`：候选 skill（作为 prompt 的 candidate 部分）
- `action`：AutoSkill 的决策（作为 reward 计算的参考）
- `label`：SkillNet 筛选的 positive/negative（作为 reward 信号）

**不需要新数据，不需要实时提取 skill，不需要实际执行 add/merge。**

### GRPO 一步的完整过程

```
Step 1: 随机采样一条 transition
  bank_snapshot = [sk_001 "python-standards", sk_002 "report-writing", sk_003 "data-analysis"]
  candidate = "code-review-policy: Enforce 2 reviewers, CI pass..."
  gt_action = "add"（AutoSkill 的决策，经 SkillNet 筛选为 positive）

Step 2: 构造 prompt（和 SFT 格式完全一样）
  Current Skill Bank (3 skills):
    [1] python-standards: Enforce type hints and docstrings...
    [2] report-writing: No tables, cite all sources...
    [3] data-analysis: Use pandas for analysis...

  Candidate Skill:
    Name: code-review-policy
    Description: Enforce 2 reviewers, CI pass, no force push
    Instructions: # Goal...

  Decide: add, merge, or discard?

Step 3: Controller 生成 G=5 个决策（temperature=0.7）
  决策 1: {"operation": "add"}
  决策 2: {"operation": "merge"}
  决策 3: {"operation": "add"}
  决策 4: {"operation": "discard"}
  决策 5: {"operation": "merge"}

Step 4: 每个决策打分（不需要实际执行）
  决策 1 (add):     和 gt_action="add" 一致 → score = 1.0
  决策 2 (merge):   不一致 → score = 0.3
  决策 3 (add):     一致 → score = 1.0
  决策 4 (discard): 不一致 → score = 0.0
  决策 5 (merge):   不一致 → score = 0.3

Step 5: Group Relative Advantage
  mean = (1.0 + 0.3 + 1.0 + 0.0 + 0.3) / 5 = 0.52
  advantage_1 (add):     1.0 - 0.52 = +0.48  ← 最高
  advantage_2 (merge):   0.3 - 0.52 = -0.22
  advantage_3 (add):     1.0 - 0.52 = +0.48
  advantage_4 (discard): 0.0 - 0.52 = -0.52  ← 最低
  advantage_5 (merge):   0.3 - 0.52 = -0.22

Step 6: Policy Update
  add 概率增大，discard 概率减小
```

### Reward 设计

**基础 reward：和 GT 决策对比**

```python
def compute_reward(decision, transition):
    gt = transition["action"]
    label = transition["label"]  # positive or negative

    if label == "positive":
        # GT 是好决策，和 GT 一致得高分
        return 1.0 if decision == gt else 0.0
    elif label == "negative":
        # GT 是坏决策，和 GT 不一致反而得高分
        return 0.0 if decision == gt else 0.5
```

**进阶 reward：结合 similarity 的细粒度打分**

```python
def compute_reward(decision, transition):
    similar = transition["similar_hits"]
    top1_sim = similar[0]["score"] if similar else 0

    if decision == "merge" and top1_sim > 0.5:
        return 1.0   # 高相似度时 merge 好
    elif decision == "add" and top1_sim < 0.3:
        return 1.0   # 低相似度时 add 好
    elif decision == "discard" and transition.get("skill_quality", {}).get("is_high_quality") == False:
        return 1.0   # 低质量 skill 被 discard 好
    elif decision == "add" and top1_sim > 0.7:
        return 0.0   # 高相似度时 add 造成冗余
    else:
        return 0.5   # 中间情况
```

### 伪代码

```python
controller = load_sft_model("qwen-1.5b-sft")
transitions = load_transitions("sft_positive.jsonl")  # Phase 1 的数据

for epoch in range(num_epochs):
    random.shuffle(transitions)
    for t in transitions:
        prompt = build_prompt(t["skill_bank_before"], t["candidate"])

        # 生成 G 个决策
        decisions = [controller.generate(prompt, temp=0.7) for _ in range(G)]

        # 打分
        scores = [compute_reward(d["operation"], t) for d in decisions]

        # GRPO update
        advantages = [s - mean(scores) for s in scores]
        controller.grpo_update(prompt, decisions, advantages)
```

### 训练配置

```
模型: SFT 训好的 Qwen2.5-1.5B
数据: Phase 1 的 transitions（~1000 条，每条采样 G=5 个决策）
GPU: 1x A800, ~4-8 小时
Epochs: 2-3
API 费用: $0（不需要 LLM call，reward 由规则计算）
```

### 和常规 GRPO 的对比

| | 常规 GRPO (math) | 我们的 GRPO |
|---|---|---|
| 每步输入 | 一道数学题（独立） | 一条 transition 的 bank+candidate（独立采样） |
| 每步输出 | 解题过程（几百 token） | add/merge/discard（几个 token） |
| reward | 答案对不对（0/1） | 和 GT 一致 + similarity 合理性 |
| bank 变化 | 无 | 无（用历史快照，不实时更新） |
| 训练稳定性 | 高 | 高（每步独立） |

### Phase 3b: 链式 GRPO（SAGE 风格）[可选进阶]

> 参考: [SAGE: Reinforcement Learning for Self-Improving Agent with Skill Library](https://arxiv.org/abs/2512.17102)

**动机**: Phase 3a（独立采样）每步独立，模型学不到"前面 add 了 3 个 coding skill，后面该 merge 而不是再 add"这种多步协同策略。链式 GRPO 让 bank 在一条 episode 里真的在变化，模型能学到 bank 演化的长期策略。

**和 SAGE 的对应关系**:

| SAGE | 我们的方法 |
|------|---------|
| Task chain（相似任务序列） | Conversation chain（WildChat 对话序列） |
| Skill library 逐步增长 | Skill bank 逐步演化 |
| Agent 在 task 中用 skill | Controller 决定 add/merge/discard |
| Skill-integrated Reward | 待定 |
| Sequential Rollout | 链式 GRPO（bank 实时更新） |

**一条 Episode**:

```
初始 bank = []
对话序列: [conv_1, conv_2, ..., conv_K]（K=20~50，从 WildChat 取连续对话）

Step 1: bank=[] + candidate_1 → Controller 生成 G 个决策
  → 选最优 → bank 更新

Step 2: bank=[sk_1] + candidate_2 → Controller 生成 G 个决策
  → 选最优 → bank 更新

...

Step K: bank=[sk_1,...,sk_N] + candidate_K → Controller 生成 G 个决策
  → 选最优 → 最终 bank

Episode Reward = eval(最终 bank) → 回传给整条链
```

**模型能学到的额外能力**（独立采样学不到的）:
- bank 小的时候激进 add，大了以后保守 merge/discard
- 同领域 skill 积累到一定数量后自动 merge
- 长期 token efficiency 的平衡

**Reward 设计**: 待定。可能的方向：
- 只看最终 bank 质量（episode-level reward）
- 每步 reward + 最终 reward 混合
- Multi-objective: performance + token efficiency + stability

**训练策略（三阶段递进）**:

```
Phase 2:  SFT         → 学格式和基本决策 pattern
Phase 3a: 独立 GRPO   → 学单步最优决策（稳定）
Phase 3b: 链式 GRPO   → 学多步协同策略（进阶，bank 演化）
```

Phase 3b 是可选的，如果 Phase 3a 效果已经够好可以跳过。

### TODO

- [ ] 实现 GRPO reward 函数（Phase 3a 用）
- [ ] 实现 GRPO 训练 pipeline（基于 TRL GRPOTrainer）
- [ ] 训练 + 评估独立 GRPO 模型（Phase 3a）
- [ ] (可选) 实现链式 GRPO episode 构造
- [ ] (可选) 设计链式 GRPO 的 reward
- [ ] (可选) 训练 + 评估链式 GRPO 模型（Phase 3b）

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
| 4 | Multi-objective: α × Δperf - β × Δtoken | ⬜ 待决策 |
| 5 | Controller 架构: 方案 B (End-to-End LM) 为主 | ✅ 倾向 |
| 6 | Base model: Qwen-1.5B vs LLaMA-3.2-1B | ⬜ 待决策 |
| 7 | 评估 benchmark: SkillsBench + ALFWorld + WebShop | ⬜ 待确认 |
