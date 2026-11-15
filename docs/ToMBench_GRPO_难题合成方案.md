# ToMBench GRPO 难题合成方案

## 结论

“提高原生出题难度、把失败机制传递到 Stage 3”是正确方向，但需要三点修正：

1. difficulty_distribution 是诊断报告中的建议分布，不能用 max() 把概率最高的标签直接当成每道题的难度。
2. 题目难度由信息结构、干扰逻辑和目标认知操作共同决定，增加句数只能作为约束之一。
3. 所有题目都锁定为 hard 会增加不可判定、外部知识依赖和多个认知操作混杂的风险。应按配置调度 hard，并对高错率维度提高 hard 比例。

本方案面向 ToMBench 的 GRPO 数据合成。目标是先生成约 2,300 条 raw 候选，经过质量过滤后保留至少 2,000 条可直接进入 tom_exact 奖励链路的 mcq_single 样本。

## 难度调度

Stage 3 应按以下优先级决定难度：

1. 数据集或运行配置中的 target_difficulty。
2. 维度报告中的 difficulty_policy。
3. 诊断报告中的 difficulty_distribution，按权重采样，而不是取最大值。
4. 没有有效配置时使用 medium。

ToMBench 的 2,300 条 raw 候选建议采用：

| 难度 | 比例 | 说明 |
|---|---:|---|
| hard | 70% | 高错率维度、需要信息隔离或语用推理的维度 |
| medium | 25% | 保持可判定性和能力连续性 |
| easy | 5% | 基础能力回放，防止奖励分布完全二值化 |

高错率维度可以将 hard 提高到 80%–90%，但仍保留少量 medium 题作为稳定锚点。target_difficulty: hard 只适用于专门的 hard 子集，不应覆盖整个 ToMBench 合成任务。

建议新增运行配置字段：

    synthesis:
      target_difficulty: hard_mix
      difficulty_weights:
        hard: 0.70
        medium: 0.25
        easy: 0.05
      hard_dimension_error_rate: 0.55
      hard_dimension_weight: 0.85
      min_hard_constraints: 3

hard_dimension_error_rate 表示维度内部错误率达到该值后，提高 hard 采样权重；它不改变报告的维度配额，只改变该维度下的题目难度调度。

## Hard 题约束

Hard 题需要满足至少三类复杂度约束，并且每条约束都必须服务于题目所声明的主认知操作。不能为了增加长度而加入无法用于判分的枝节。

### 信息结构约束

至少包含以下两项：

- 多阶段信息变化：至少两次状态或信息更新，且存在角色离场、未在场或只收到转述的情况。
- 明确的信息边界：故事要交代谁看到、谁没看到、谁只听到部分内容。
- 目标角色的可见信息与全局真实状态不同。
- 若考察二阶信念，必须明确区分 A 以为 B 知道什么和 B 实际知道什么。

### 干扰结构约束

至少包含一项真实但不决定答案的事件，例如：

- 另一角色发生情绪变化，但不影响目标角色的信念。
- 一个看似关键的物品或行动实际只是背景事件。
- 表面上支持错误答案的线索，后续被信息来源或时间顺序削弱。

干扰事件必须可以被审查者指出其作用，不能用随机细节充当 red herring。

### 认知操作约束

至少包含一项目标操作，并在元数据中记录：

- 视角隔离：不能用全局事实替代目标角色的局部信念。
- 二阶或更高阶信念：问题指向 A 以为 B 以为……，不能退化成一阶事实问答。
- 隐式语用：字面礼貌、安慰、反讽、试探或回避与真实意图不一致。
- 基于错误信念的下一步行动：行动必须由角色当前相信的内容推出。
- 时间和因果绑定：必须区分决策时信息与事后真相。

### 判分约束

Hard 题必须仍然满足以下条件：

- 正确答案由故事内信息唯一支持。
- 至少一个干扰项对应模型常见的 shortcut。
- 至少一个干扰项对应全局真实状态、其他角色信念或较浅推理层级。
- 不需要外部常识、文化背景或专业知识才能排除干扰项。
- 题目只考察一个主认知操作；其他操作只能作为辅助约束。

## ToMBench 的数据集提示改造

ToMBench 的 complexity_hint 应扩展为可检查的生成契约：

- Hard 题使用 5–8 个自然段或编号句，至少 3 个角色，其中至少 1 个角色在关键状态变化时不在场。
- 事件链至少包含初始状态、一次未观察变化、一次转述或误导、一个与问题无关但真实发生的干扰事件。
- 问题必须明确目标角色和目标时刻，避免让评测者在多个时间点之间猜测。
- 选项应分别覆盖：目标角色真实信念、全局事实、另一角色信念、浅一阶答案或字面语用答案。
- 生成器必须在内部自检后再输出，最终 JSON 不得包含思维过程、诊断字段或答案解释。

增加句数不能替代认知约束。若 5–8 句中只有一条信息变化，仍然只能算 medium。

## Stage 2 诊断报告扩展

DimensionDiagnosisReport 应增加以下字段，使用列表以支持多个失败机制，并为旧报告提供空列表默认值：

    failure_trigger: list[str] = Field(default_factory=list)
    required_distractor_logic: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    difficulty_policy: str = Field(default="weighted")

字段语义：

- failure_trigger：脱敏后的失败触发器，描述模型把什么信息、时间点或角色视角混在了一起。例如“把全局事实当成 Agent A 的私密信念”。不得复制原题中的人名、地点、物品和完整句子。
- required_distractor_logic：本维度必须出现的干扰项逻辑，例如“一个选项必须是全局真实状态，另一个必须是另一角色所相信的状态”。
- hard_constraints：该维度生成 hard 题时必须满足的结构约束，例如“加入一次目标角色未在场的状态变化”。
- difficulty_policy：允许 weighted、hard_mix、hard_only。默认使用 weighted，高错率维度使用 hard_mix。

Stage 2 的诊断 prompt 需要要求模型输出这些字段，并明确禁止把原题文本直接复制到诊断报告中。诊断失败时的 fallback 报告也要填入保守的失败触发器、干扰项逻辑和 hard 约束。

## Stage 3 的报告透传

Stage 3 的生成 prompt 应同时透传：

- common_error_patterns
- failure_trigger
- required_distractor_logic
- hard_constraints
- primary_cognitive_operation
- 当前变体的 difficulty policy

生成任务应从“生成该维度的一道题”改成“构造一个复现失败机制、但不复用原题表面内容的新场景”。生成器需要完成以下自检：

1. 目标角色在问题所指时间点能获得哪些信息？
2. 哪个信息是全局可见但目标角色不可见的？
3. red herring 为什么不会改变正确答案？
4. 每个干扰项对应哪个 shortcut？
5. 删除任意一个 hard 约束后，题目是否会变简单或变成不可判定？

如果第 3 或第 4 项无法回答，候选题应进入 repair，不应直接进入 clean 数据。

## 2000 条 ToMBench GRPO 数据路径

以 checkpoint-124 的 ToMBench 错题为输入，使用独立 run 目录，不覆盖既有 feedback_output：

    stage: all
    api_profile: synthesis
    output_path: <new-run>/feedback_output

    synthesis:
      bad_cases_per_report: 2
      diagnosis_batch_size: 12
      synthesis_batch_size: 20
      max_retries_per_diagnosis: 3
      target_difficulty: hard_mix

    synthesis_datasets:
      - name: ToMBench
        target_samples: 2300
        samples_per_report: 5

target_samples: 2300 是 raw 目标，目的是过滤后保留至少 2,000 条 clean 样本。460 份报告、每份 5 个变体能提供足够的故事变化；bad_cases_per_report: 2 避免当前 1,135 道错题在不放回采样时提前耗尽。

raw 候选必须依次通过：

1. ToMBench schema 校验：4 个唯一选项、一个正确答案、正确字母在 A–D 内。
2. 嵌入选项剥离检查：question 不得再次包含第二套 A/B/C/D 选项。
3. answerability 检查：故事内信息足以唯一决定正确答案。
4. shortcut 检查：至少一个干扰项对应诊断报告要求的失败机制。
5. 测试集泄漏和内部 MinHash 去重。
6. 维度覆盖和 hard 约束统计。

最终 ROLL 导出使用 single:A/B/C/D、tom_exact tag 和 tom_exact domain。训练前保留约 10% 的新样本作为内部验证，并加入少量已有高质量 ToMBench replay，防止模型只适应合成题表面模式。

## 验收标准

本轮数据进入 GRPO 前必须满足：

- raw 候选约 2,300 条，clean 样本不少于 2,000 条。
- hard 占比达到 70% 左右，高错率维度达到 80%–90%。
- 每个活跃 ToMBench 能力维度至少保留 20 条 clean 样本，或在报告中记录无法满足的原因。
- 每条 hard 题至少满足 3 类复杂度约束。
- failure_trigger 和 required_distractor_logic 非空的维度，其生成题必须能在审计结果中找到对应证据。
- 选项重复、选项内嵌双渲染、答案不唯一、故事依赖外部知识的样本全部隔离。
- 先用 400 条 clean 样本进行 GRPO pilot，再决定是否启动完整 2,000 条训练。

该方案保留 feedback 的原生诊断、合成和过滤链路，同时把难度调度和失败机制复现变成可审计字段，避免单纯增加长度后仍然生成容易题或不可判定题。
