# Findings & Research Notes

## 2026-03-17: Training-Free GRPO 分析

### 核心发现：Controller 不稳定性的根源

通过分析 youtu-agent 的 `ExperienceUpdater` 实现，确认以下问题：

1. **无优化目标**: Controller（`_group_update` + `_batch_update`）的 ADD/DELETE/UPDATE/NONE 决策完全由 LLM prompt call 驱动，没有 loss function，没有 learning rate，没有收敛保证。

2. **误差累积无纠正**: 每个 batch 的决策不可逆地叠加。如果某步 UPDATE 改坏了一条经验，后续基于错误经验继续决策，误差链式传播。代码中没有 rollback 机制。

3. **评估与决策解耦**: `practice()` 中 eval 是可选的（`do_eval: false` 是默认值），且即使开启，eval 结果也不影响 Controller 的决策。Controller 不知道自己的上一轮更新是好是坏。

4. **两次 LLM 调用目标不对齐**: Group Computation（从轨迹对比提取 insight）目标明确；Controller（维护高质量经验池）目标模糊，"高质量"的标准完全靠自然语言 prompt 描述。

### youtu-agent 架构关键接口

**可复用组件**:
- `RolloutManager(BaseBenchmark)`: 批量 rollout 执行，支持 preprocess → rollout → judge → stat 四阶段 pipeline
- `EvaluationSample`: 数据容器，生命周期 init → rollout → judged
- `SimplifiedAsyncOpenAI`: 异步 LLM 客户端
- `ExperienceCache`: SQLModel 数据库缓存
- `ConfigLoader`: Hydra 配置加载

**需替换组件**:
- `ExperienceUpdater._group_update()` + `_batch_update()` → TrainedController
- `TaskRecorder.experiences: dict[str, str]` (flat) → HierarchicalSkillBank (tree)

### Training-Free GRPO 数据流（完整）

```
TrainingFreeGRPO.practice()
  for epoch:
    load_epoch_data(epoch, shuffle, truncate)
    # → 每条 query 复制 grpo_n 次

    for batch:
      # 1. Rollout
      RolloutManager.main(batch_idx)
        preprocess_batch()  → 数据预处理
        rollout_batch()     → agent 执行（并发，带 timeout/retry）
        judge_batch()       → 验证答案正确性
        stat_batch()        → 统计指标
      # 输出: List[EvaluationSample] with reward

      # 2. Experience Update (= Controller)
      ExperienceUpdater.run(rollouts)
        _single_rollout_summary()  → 单轨迹摘要（筛选 0<avg<1 的 group）
        _group_advantage()         → 组内对比 → 提取 A_text
        _group_update()            → 每组: A_text vs 经验池 → operations
        _batch_update()            → 全 batch operations 合并 → apply
      # 输出: 更新后的 experience dict

      # 3. Cache
      ExperienceCache.save_experiences()

      # 4. Optional Eval
      eval_rollout_manager.main()
```

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
