# Progress Log

## Session 1: 2026-03-17 — Problem Analysis & Idea Formation

### Completed
- [x] 分析 Training-Free GRPO 的完整代码实现
- [x] 深入理解 ExperienceUpdater 的 4 阶段流程
- [x] 识别 Controller 不稳定性的 4 个根源
- [x] 确定研究定位和两个核心主张
- [x] 讨论并选择叙事方向
- [x] 讨论训练 Controller 的可行性

## Session 2: 2026-03-17 — Phase 1 Implementation (skillcontroller)

### Completed
- [x] 搭建 skillcontroller 项目基础结构
- [x] 实现 InstrumentedGRPO wrapper
- [x] 实现 StabilityMetrics + 分析脚本
- [x] 测试代码（test_skill_bank, test_stability）

## Session 3: 2026-03-17~18 — GPU 服务器环境 + Training-Free GRPO

### Completed
- [x] GPU 服务器环境搭建（youtu conda env）
- [x] 修复 ipython/matplotlib 依赖、URL 路径、HF 镜像
- [x] 数据集加载到 SQLite
- [x] Training-Free GRPO 原版跑通（math_reasoning）
- [x] Instrumented GRPO 代码修复（patched_batch_update kwargs）
- [x] Instrumented GRPO 在 GPU 服务器上运行（遇到 OOM/API 额度问题）

## Session 4: 2026-03-18~19 — AutoSkill 数据收集 Pipeline

### Completed
- [x] 探索 AutoSkill 代码架构（client.py, maintenance.py, extraction.py）
- [x] 实现 `skillcontroller_pipeline/` 完整 pipeline
  - `instrumented_sdk.py` — InstrumentedAutoSkill wrapper
  - `feature_extractor.py` — 17 维 domain-agnostic 特征
  - `data_converter.py` — MLP + LM 两种训练数据格式
  - `scripts/prepare_wildchat.py` — WildChat 下载 + 筛选
  - `scripts/collect_autoskill_data.py` — 批量收集 transitions
  - `scripts/convert_to_training_data.py` — 格式转换
- [x] 修复 AutoSkill generic provider 的 base_url 问题
- [x] 在 GPU 服务器上跑通 pipeline（5 条 transition 验证成功）
  - 3 个 add + 2 个 merge，数据格式正确
- [x] 确定训练策略：SFT warmup + GRPO refinement（非 DPO）

### Key Decisions Made
1. **训练策略**: SFT（AutoSkill 数据）+ GRPO（自我改进），不用 DPO
2. **数据收集主力**: AutoSkill（快、便宜、通用），Training-Free GRPO 作为补充待定
3. **Controller 架构**: 方案 A (LLM+MLP) 和方案 B (End-to-End LM) 都做，实验选最佳，待定
4. **Multi-objective**: reward = α × Δperf - β × Δtoken，待定

### Current Status

| 组件 | 状态 |
|------|------|
| skillcontroller 项目结构 | ✅ Done |
| InstrumentedGRPO (Training-Free GRPO) | ✅ 代码完成，GPU 上运行中，后续可能用不上 |
| AutoSkill pipeline | ✅ 代码完成，已跑通验证 |
| SFT 训练数据（AutoSkill） | ⬜ 需要加大数据量 |
| GRPO 训练 pipeline | ⬜ 待实现 |
| Controller 模型 | ⬜ 待实现 |
| 评估 | ⬜ 待实现 |

### Next Steps
- [ ] 调研哪些数据用来产生sft数据，刷什么benchmark（极度重要）
- [ ] 充值 API → 跑 2000 条 WildChat × 3 runs 收集 AutoSkill transition
- [ ] 实现合成数据（rewrite math/coding/writing 规范为对话格式）（暂时取消）
- [ ] 运行 convert_to_training_data.py 产出 SFT 训练数据
- [ ] 实现 SFT 训练 pipeline
- [ ] 实现 LLM-as-Judge eval（GRPO 需要）
- [ ] 实现 GRPO 训练 pipeline
