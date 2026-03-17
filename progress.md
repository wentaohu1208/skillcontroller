# Progress Log

## Session 1: 2026-03-17 — Project Initialization & Problem Analysis

### Completed
- [x] 分析 Training-Free GRPO 的完整代码实现
- [x] 识别 Controller 不稳定性的 4 个根源
- [x] 确定研究定位和两个核心主张
- [x] 创建项目规划文件

## Session 2: 2026-03-17 — Phase 1 Implementation

### Completed

**1.1 项目基础结构** ✅
- `src/config/` — SkillControllerConfig, SkillBankConfig, ControllerConfig, DataCollectionConfig
- `src/skill_bank/` — SkillNode (3-level enum), HierarchicalSkillBank (CRUD/snapshot/restore/serialize)
- `src/controller/` — BaseController, ControllerAction, ControllerInput/Output, OperationType
- `src/utils/` — set_seed, get_logger

**1.2 youtu-agent 集成 (数据收集 pipeline)** ✅
- `src/data_collection/instrumented_grpo.py` — InstrumentedGRPO
  - 继承 TrainingFreeGRPO 的 practice() 逻辑
  - 在 experience update 前后 snapshot
  - Monkey-patch _batch_update 捕获 operations
  - 可选 held-out eval 计算 delta_reward
  - 每步输出 TransitionRecord
- `src/data_collection/recorder.py` — TransitionRecorder + TransitionRecord
- `scripts/collect_data.py` — 完整的数据收集入口脚本

**1.3 稳定性度量** ✅
- `src/data_collection/stability.py` — StabilityMetrics
  - rollback_rate: 更新导致 reward 下降的比例
  - skill_churn: ADD+DELETE 操作占比
  - skill_lifecycle: 每条经验的生存周期追踪
  - reward/bank_size trajectory

**1.5 分析脚本** ✅
- `scripts/analyze_stability.py` — 完整分析 pipeline
  - 基础稳定性指标
  - Operation-reward 相关性分析（哪种操作导致 reward 下降）
  - Bank 增长分析
  - Experience 波动性分析（最不稳定的经验）
  - 关键发现自动总结

**Tests** ✅
- `tests/test_skill_bank.py` — SkillBank CRUD/snapshot/serialize 测试
- `tests/test_stability.py` — StabilityMetrics 测试

### Phase 1 Status

| Task | Status |
|------|--------|
| 1.1 项目基础结构 | ✅ Done |
| 1.2 youtu-agent 集成 | ✅ Done |
| 1.3 稳定性度量 | ✅ Done |
| 1.4 跑 baseline 收集数据 | ⬜ 需要运行 collect_data.py |
| 1.5 数据分析 | ✅ 脚本完成，待数据 |

### How to Run

```bash
# Step 1: 收集数据（在 youtu-agent 环境中）
cd /Users/wentaohu/project/skillcontroller
python scripts/collect_data.py \
    --config_name math_reasoning \
    --experiment_name math_stability_v1 \
    --save_dir data/collected \
    --held_out_eval \
    --held_out_size 20

# Step 2: 分析稳定性
python scripts/analyze_stability.py \
    --data_path data/collected/math_stability_v1_transitions.json \
    --output_dir data/analysis
```

### Next Steps
- [ ] 运行 collect_data.py 收集 math 任务的 transition 数据
- [ ] 运行 analyze_stability.py 生成稳定性分析报告
- [ ] 根据分析结果确定 Phase 2/3 的具体设计方向
