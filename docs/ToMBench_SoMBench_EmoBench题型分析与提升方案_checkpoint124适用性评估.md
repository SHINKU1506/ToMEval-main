# ToMBench、SoMBench、EmoBench 方案对 checkpoint-124 的适用性评估

## 结论

原题型分析对当前 checkpoint-124 仍有较高参考价值，但应作为“定向数据和 GRPO 采样策略”使用，不能把其中上一轮 SFT 的分数、失分排序和训练集分布直接当作当前事实。

当前建议：

- 保留 ToMBench / SoMBench / EmoBench 的弱项定向合成方向。
- 保留 SoMBench Q2、Q3、varA/varB 的成组对照设计。
- 保留 EmoBench emotion、persona、Response、Action 和 Personal-Others 的补强方向。
- 暂缓把 Scalar-Implicature、Persuasion 等上一轮低分任务直接设置成最高配额，先从当前 prediction 重新统计 task 级错误。
- “97.6% answer-only 导致 CoT 失调”和“GRPO 中 grouped 题型为 0”需要核对 checkpoint-124 的实际训练 manifest 后再作为硬结论。

## 当前结果与旧方案对照

| 数据集 | 旧方案记录的 SFT 分数 | checkpoint-124 当前结果 | 判断 |
|---|---:|---:|---|
| ToMBench | 80.16% | 83.04% | 当前明显高于旧记录，旧分数不再适用 |
| SoMBench 选择题 | 67.20% | 70.40% | 当前明显高于旧记录；Q2/Q3 方向仍适用 |
| EmoBench | 68.50% | 73.13% | 当前明显高于旧记录；emotion、Response、Action 仍是主要失分源 |

当前评测数据：

- ToMBench：5,720 条，4,750 正确，83.04%。
- SoMBench：实际生成 3,481 条；其中 3,000 道选择题规则准确率 70.40%，481 道开放题未做 API judge。
- EmoBench：800 条，585 正确，73.13%，抽取失败 0。
- EmoBench pass@8：215 道基线失分题中恢复 87 道，恢复率 40.47%；反事实全量上限 84.00%。

## 仍然适用的部分

### ToMBench

当前失分量最高的能力为：

- Faux pas：221 条
- Knowledge-Information links：155 条
- Desire-action：70 条
- Atypical emotional reactions：69 条
- Intentions explanations：51 条

因此原方案中 Faux-pas、知识与注意链接、欲望/行动和情绪相关能力仍应优先保留。Scalar-Implicature、Persuasion、Unexpected-Outcome 在旧评测中很弱，但当前报告尚未重新输出 task 级统计，需先复算后决定配额。

### SoMBench

当前规则可计分的 3,000 道选择题中：

- Q3 失分 564 条
- Q2 失分 212 条
- Q1 失分 112 条
- Q2 多选准确率 62.85%
- Q1/Q3 合并的 mcq_single 准确率 71.93%

Q3 和 Q2 仍是明确主攻方向。原方案中的“无法确定”偏置需要重新生成当前混淆矩阵后确认，不能只沿用旧数字。

### EmoBench

当前主要失分源：

- emotion_recognition：100 条
- Response：59 条
- Action：56 条
- perspective_taking：37 条
- personal_beliefs_and_experiences：36 条
- persona：55.88%，但样本只有 34 条
- cultural_value：69.57%

原因判断 cause 为 88.75%，情绪识别 emotion 为 77.50%，说明当前优先级应是“情绪标签精判和人物状态”，而不是继续大量生成简单的原因题。Grouped 题整体为 75.00%，仍需保留 grouped 数据用于 GRPO，但当前结果不能证明训练集里 grouped 已经覆盖充分。

## 需要修正的旧结论

1. 旧方案的 80.16%、67.20%、68.50% 是上一轮 SFT 结果，当前 checkpoint 已分别提升到 83.04%、70.40%、73.13%。
2. 旧方案中的 ToMBench task 级失分排序不能直接覆盖当前结果；当前已确认的高失分能力应优先使用。
3. 旧方案关于 GRPO 数据中 HiToM 占 61.5%、三类 benchmark 合计约 18%、EmoBench grouped 为 0 条，必须通过当前 GRPO manifest 重新核验。
4. 旧方案建议“禁止继续使用高比例 answer-only 做 SFT”适用于下一轮 SFT 设计；当前 checkpoint 已经训练完成，当前主任务应转向 GRPO 数据构建和 reward 验证。
5. EmoBench 元数据中的 400 条 unknown 不能解释为真实能力类别，当前弱项分析应使用 dimension、subset、question_subtype 和已有细粒度字段。

## 对当前 GRPO 的执行建议

第一版数据预算建议：

- ToMBench 25%：Faux pas、Knowledge-Information links、Desire-action、Atypical emotional reactions，并补算当前 task 级错误后再增加 Scalar-Implicature/Persuasion。
- SoMBench 25%：Q3 50%、Q2 30%、Q1 hard/varA/varB 20%。Q3 先做“是/否/无法确定”三联对照，Q2 逐项核验选项证据。
- EmoBench 15%：emotion_recognition 40%、Response 25%、Action 20%、persona/cultural_value/Personal-Others 15%；保留 EU grouped 双问。
- HiToM 15%：只保留高阶和当前模型有 reward 区分度的样本。
- 控制样本 20%：用于检测训练后是否出现回退。

每条候选进入 GRPO 前仍需经过结构检查、answerability、shortcut、LSH 和 group-level reward variance。pass@8 能恢复 87 道失分题说明部分错题具有可学习 reward 差异；128 道 8 次均失败的题应进入修复或 hard-negative 池，不应直接占据主训练集。

## 文件来源

- 原始方案：ToMBench_SoMBench_EmoBench题型分析与提升方案.md
- 当前 EmoBench：EmoBench_SFT_r13_checkpoint124_评测结果.md
- 当前离线汇总：D:\workspace\pre\rsi\output\tomeval-sft-r13-20261114\offline_summary\SUMMARY.md
- 当前 ToMBench 预测：D:\workspace\pre\rsi\output\tomeval-sft-r13-20261114\retrieved\ToMBench\prediction.jsonl
- 当前 SoMBench 预测：D:\workspace\pre\rsi\output\tomeval-sft-r13-20261114\retrieved\SoMBench\prediction.jsonl
