# Progress Log

## Session 1: 2026-03-17 — Problem Analysis & Idea Formation

### Completed
- [x] 分析 Training-Free GRPO 的完整代码实现（youtu-agent/utu/practice/）
- [x] 深入理解 ExperienceUpdater 的 4 阶段流程
- [x] 识别 Controller 不稳定性的 4 个根源
- [x] 确定研究定位：独立的 skill bank management 问题，非 Training-Free GRPO 改进
- [x] 确定两个核心主张：Trained Controller + Hierarchical Skill Bank
- [x] 讨论并选择叙事方向："Learning to Maintain Hierarchical Skill Banks for LLM Agents"
- [x] 讨论训练 Controller 的可行性，确定跨任务离线训练路线

## Session 2: 2026-03-17 — Phase 1 Implementation

### Completed
- [x] 1.1 搭建项目基础结构
  - `src/config/` — SkillControllerConfig, SkillBankConfig, ControllerConfig, DataCollectionConfig
  - `src/skill_bank/` — SkillNode (3-level enum), HierarchicalSkillBank (CRUD/snapshot/restore/serialize)
  - `src/controller/` — BaseController, ControllerAction, ControllerInput/Output, OperationType
  - `src/utils/` — set_seed, get_logger
- [x] 1.2 youtu-agent 集成
  - `src/data_collection/instrumented_grpo.py` — InstrumentedGRPO wrapper
  - `src/data_collection/recorder.py` — TransitionRecorder + TransitionRecord
  - `scripts/collect_data.py` — 数据收集入口脚本
- [x] 1.3 稳定性度量
  - `src/data_collection/stability.py` — StabilityMetrics (rollback_rate, churn, lifecycle)
- [x] 1.5 分析脚本
  - `scripts/analyze_stability.py` — operation-reward 相关性、bank 增长、波动性分析
- [x] Tests — test_skill_bank.py, test_stability.py

## Session 3: 2026-03-17 — GPU 服务器环境配置

### Completed
- [x] GPU 服务器 (`/data/hwt/youtu-agent/`) 环境搭建
- [x] 安装缺失依赖: ipython, matplotlib, math-verify
- [x] 配置 .env: DeepSeek-chat via 中转站 `api.qingyuntop.top/v1`
- [x] 修复 `UTU_LLM_BASE_URL` 路径重复问题（不要带 `/chat/completions`）
- [x] 设置 HuggingFace 镜像（`HF_ENDPOINT=https://hf-mirror.com`）
- [x] 数据集加载到 SQLite DB（AIME24/25, DAPO-Math-17k, AFM_web_RL, WebWalkerQA）
- [x] 开始跑 baseline 评估（`python scripts/run_eval.py --config_name math/math_AIME24`）

### Phase 1 Status

| Task | Status |
|------|--------|
| 1.1 项目基础结构 | ✅ Done |
| 1.2 youtu-agent 集成 | ✅ Done |
| 1.3 稳定性度量 | ✅ Done |
| 1.4 Baseline 评估 | 🔄 运行中 (GPU 服务器) |
| 1.5 跑 Training-Free GRPO 原版 | ⬜ Baseline 完成后进行 |
| 1.6 跑 Instrumented 版本收集 transition | ⬜ 待 1.5 验证通过 |
| 1.7 稳定性分析 | ⬜ 待数据 |

### Key Decisions Made
1. **任务选择**: Math (AIME24/DAPO-Math-17k)
2. **项目关系**: 独立项目，sys.path 引用 youtu-agent
3. **叙事方向**: 独立问题 "skill bank management"，不是 GRPO 改进
4. **训练路线**: 跨任务离线训练

### Next Steps
- [ ] 等 baseline 评估完成，记录分数
- [ ] 跑 Training-Free GRPO 原版（`python scripts/run_training_free_GRPO.py --config_name math_reasoning`）
- [ ] 把 skillcontroller 代码部署到 GPU 服务器
- [ ] 跑 Instrumented 版本收集 transition 数据
- [ ] 运行 analyze_stability.py 分析
- [ ] 根据分析结果推进 Phase 2/3
