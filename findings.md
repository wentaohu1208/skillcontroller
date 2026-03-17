# Findings & Research Notes

## 2026-03-17: Training-Free GRPO 分析

### 核心发现：Controller 不稳定性的根源

通过分析 youtu-agent 的 `ExperienceUpdater` 实现，确认以下问题：

1. **无优化目标**: Controller（`_group_update` + `_batch_update`）的 ADD/DELETE/UPDATE/NONE 决策完全由 LLM prompt call 驱动，没有 loss function，没有 learning rate，没有收敛保证。

2. **误差累积无纠正**: 每个 batch 的决策不可逆地叠加。如果某步 UPDATE 改坏了一条经验，后续基于错误经验继续决策，误差链式传播。代码中没有 rollback 机制。

3. **评估与决策解耦**: `practice()` 中 eval 是可选的（`do_eval: false` 是默认值），且即使开启，eval 结果也不影响 Controller 的决策。Controller 不知道自己的上一轮更新是好是坏。

4. **两次 LLM 调用目标不对齐**: Group Computation（从轨迹对比提取 insight）目标明确；Controller（维护高质量经验池）目标模糊，"高质量"的标准完全靠自然语言 prompt 描述。

### Training-Free GRPO 的 4 阶段 Experience Update 流程

```
ExperienceUpdater.run(rollouts):
  Stage 1: _single_rollout_summary()
    - 对每条轨迹做结构化摘要
    - 只处理 "部分正确" 的 group (0 < avg_reward < 1)

  Stage 2: _group_advantage()
    - 同一 query 的多条摘要对比（好 vs 差）
    - 输出: A_text (语义版 group relative advantage)

  Stage 3: _group_update()          ← Controller 第一步
    - 每组 A_text vs 当前经验池
    - LLM 决定 ADD/UPDATE/DELETE/NONE
    - 输出: 每组的 operations

  Stage 4: _batch_update()          ← Controller 第二步
    - 汇总全 batch 的 operations
    - LLM 合并冲突、去重
    - Apply: 执行 ADD/UPDATE/DELETE
    - 输出: 更新后的经验池
```

Stage 1-2 是 "Group Computation"（信号提取），Stage 3-4 是 "Controller"（经验池管理）。
我们的工作替换的是 Stage 3-4。

### youtu-agent 架构关键接口

**可复用组件**:
- `RolloutManager(BaseBenchmark)`: 批量 rollout 执行，支持 preprocess → rollout → judge → stat
- `EvaluationSample`: 数据容器，生命周期 init → rollout → judged
- `SimplifiedAsyncOpenAI`: 异步 LLM 客户端
- `ExperienceCache`: SQLModel 数据库缓存
- `ConfigLoader`: Hydra 配置加载（config_path 相对于 utu/config/loader.py）

**需替换组件**:
- `ExperienceUpdater._group_update()` + `_batch_update()` → TrainedController
- `TaskRecorder.experiences: dict[str, str]` (flat) → HierarchicalSkillBank (tree)

---

## 2026-03-17: youtu-agent 运行环境记录

### 环境配置

- **GPU 服务器路径**: `/data/hwt/youtu-agent/`
- **Python 环境**: `youtu` conda env, Python 3.10
- **依赖安装**: `uv sync --all-extras` + `pip install ipython matplotlib math-verify`
- **LLM API**: DeepSeek-chat，通过中转站 `api.qingyuntop.top/v1`
- **注意事项**:
  - `UTU_LLM_BASE_URL` 只设到 `/v1`，不要带 `/chat/completions`（框架自动拼接）
  - HuggingFace 数据集需要设置镜像: `export HF_ENDPOINT=https://hf-mirror.com`
  - 数据存储在 SQLite (`test.db`)，不是本地文件

### .env 配置

```bash
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.qingyuntop.top/v1
UTU_LLM_API_KEY=<key>
UTU_DB_URL=sqlite:///test.db
```

### 已加载数据集

通过 `python scripts/data/process_training_free_GRPO_data.py` 加载到 DB:
- AIME24: AIME 2024 竞赛题
- AIME25: AIME 2025 竞赛题
- DAPO-Math-17k: 17k 数学题（Training-Free GRPO 的训练数据）
- AFM_web_RL: Web agent RL 数据
- WebWalkerQA: Web 导航问答

---

## 2026-03-17: 相关工作初步调研

### Skill/Experience Management in LLM Agents

| 方法 | Skill 表示 | 更新机制 | 层级 | 训练 |
|------|-----------|---------|------|------|
| Training-Free GRPO | flat dict | LLM prompt call | No | No |
| Voyager (2023) | code skill library | LLM 生成 + 验证 | No | No |
| Reflexion (2023) | text memory | LLM 反思 | No | No |
| ExpeL (2023) | experience pool | LLM 提取 + 累积 | No | No |
| OMNI-EPIC (2024) | skill tree | 进化式生长 | Yes | No |
| **Ours (proposed)** | hierarchical tree | trained controller | Yes | Yes |

### 关键差异点
- 现有方法全部用 LLM 做经验管理，没有一个用 trained model
- OMNI-EPIC 有层级但是进化式（随机变异+选择），不是 learned
- 我们的独特贡献: **learned controller + hierarchical structure**

### 待深入调研
- [ ] STILL (Skill Library) 系列工作
- [ ] MemoryBank, MemGPT 等 memory management 工作
- [ ] Meta-learning for knowledge management
- [ ] Curriculum learning 与 skill bank 的关系
