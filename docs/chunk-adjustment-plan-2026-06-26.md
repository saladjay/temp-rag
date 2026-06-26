# Chunk 调整方案（基于 prompt-v2 符合度评测）

- 日期：2026-06-26
- 依据（三方对齐）：
  - `docs/eval-2026-06-26-prompt2-conformance.md` §4 残留问题（v2 prompt 后，Qwen3 53% / v4 68% 符合率）
  - `_conf.json` bench 的 retrieve/rerank top3 证据
  - `docs/chunk_audit_report.md` 碎片病理（ops FAQ 拆开 / regulation-template 过碎 / policy 标题-目录碎片）
- 前提：科小星 v2 prompt 已上线，prompt 不再是瓶颈；剩余提升空间在**召回/切块**。

---

## 1. 归因：哪些是 chunk 问题

| 题 | 现象 | 根因 | chunk？ |
|---|---|---|---|
| q26-28 待办点不动 | 漏关键步"非正常流程→搜索【处理】"，只给 2/4 原因 | ops 操作步骤/troubleshooting 被切碎，关键步召回靠后 | ✅ |
| q32/q33 平台卡/成果录入 | 双兜底，参考答案有内容 | ops FAQ 问答拆开（问题与答案分块）+ 口语查询 embedding 不命中 | ✅ |
| q17 漏论文版面费标准 | 漏项 | template 表单字段碎片化，"版面费 2000~8000元/篇"没成可召回块 | ✅ |
| q3 国家层面写成广西 | 张冠李戴 | 召回"多省印发"文档 + `规划计划_index.md` 碎片占名额；模型误归属 | ✅（消碎片）+ 轻 prompt |
| q35 v4 判错"不能过" | 审核类偏绝对 | — | ❌ prompt（审核留余地）|
| q10 Qwen3 空答 | len=0 | 生成瞬时异常 | ❌ 重试机制 |
| q2/q30 v4 兜底 | v4 偏严 | 检索阈值/上下文利用 | ❌ prompt/阈值 |

→ chunk 调整治 **q3 / q17 / q26-28 / q32 / q33**（约 8-10 题），是符合率从 53%→? 的提升空间。

---

## 2. 分级方案（对应 `chunk_audit_report.md` 的 O 编号）

### P0 — 收益最大、最具体
**P0-1. ops FAQ「问+答」合并（O7）** — 治 q26-28 / q32 / q33
- 现状：FAQ 切成"问题单块、答案单块"（审计 P2 实证：`1.账号登录，怎么办？ P2~3` 17 字单成块）。
- 改：识别 `数字.问题？` 模式 + 紧随答案段，合成「问+答」一个 chunk。命中问题文本即拿到答案。troubleshooting/操作步骤序列作为完整步骤块保留。

**P0-2. 剥 Word-TOC/目录 + 标题并入正文（O1+O4）** — 治 q3 + policy 噪声
- 现状：`规划计划_index.md` 等索引/目录碎片占召回名额（q3 top3 有它）；裸标题 `五、保障措施`(6字) 单成块。
- 改：入库前剥 `TOC \o`/`HYPERLINK`/`PAGEREF`/目录行；标题强制并入后续正文，禁 heading-only chunk。

### P1 — 治漏项
**P1-3. 表格/表单字段保全（O2）** — 治 q17（论文版面费）
- 连续 `|...|` 表格行整体一块；字段标签与其值/标准合成一块。

**P1-4. 全局小-chunk 合并器（O5）** — 普治过碎
- 切块后 post-process，<80 字 chunk 并入相邻（不论结构边界）。抬高信息密度，减 rerank 噪声。

### P2 — 数据层
**P2-5. 修 kb_policy_group 0 文件（O-dedup）** — 治"集团层面"召回（q3/q22）
- 排查 `集团规划` 去重主 path，让集团级文件入库。

### 配套（必须）
- **O13 chunk 质量报告**：改前基线 → 改后对比。
- **重灌 + 重跑 bench**：chunk 变 → segment_id 变 → 按文档幂等重灌受影响 KB（`MilvusWriter` 按 doc_id 删旧再插）→ 重跑 38 题比符合率。

---

## 3. 落地顺序
1. O13 出基线（每库尺寸/碎片率）。
2. **P0-1 + P0-2** 改 `app/ingest/chunker.py`（影响 ops + policy_national，最大两库）。
3. 重灌 ops + policy_national。
4. 重跑 38 题 bench，看 q3/q17/q26-33 符合率变化。
5. 据效果决定 P1/P2。

## 4. 非 chunk 项（另开，不并入本方案）
- q35 → prompt 加"审核类结论留余地"。
- q10 → 生成失败重试（1-2 次退避）。
- q2/q30 → prompt 强化"检索部分相关即作答"或降拒答阈值。
- 口语查询（q32"怎么这么卡"）→ 可选 rewrite 扩同义词（P0-1 FAQ 合并已能大幅缓解）。
