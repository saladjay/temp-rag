# 知识库 Chunking 审计报告

- 日期：2026-06-26
- 范围：除 kb_project 外的 5 个知识库（policy_national / policy_group / ops / regulation / template）的 **structural chunker**（`app/ingest/chunker.py::chunk_document` 结构感知路径）切块合理性
- 方法：用 kbmap manifest（已 content-hash 去重）映射真实文件 → 对每个 `.md` 跑 `chunk_document`（非 kb_project 走结构路径）→ 按 KB 汇总 chunk 字数分布、超限/过碎计数、整篇文件数 + 抽样极碎 chunk
- 护栏基准：`chunk_target_max=1500 / chunk_target_min=200 / chunk_whole_doc_max=800`

---

## 1. 数据（真实文件）

| 知识库 | 文件 | chunks | 字数 p50 | <200字 | <50字 | <30字 | p90 | 评估 |
|---|---|---|---|---|---|---|---|---|
| kb_policy_national | 169 | 3366 | 194 | 51% | 20% | 549 | 1039 | ⚠️ 偏碎（标题/目录碎片） |
| kb_ops | 16 | 423 | 224 | 49% | 31% | 88 | 1394 | ⚠️ FAQ 问与答被拆开 |
| kb_regulation | 12 | 421 | **75** | **82%** | 31% | 62 | 398 | 🔴 严重过碎 |
| kb_template | 2 | 80 | **31** | **88%** | 64% | 40 | 238 | 🔴 严重过碎（已知表单弱点） |
| kb_policy_group | 0 | — | — | — | — | — | — | ⚠️ 当前 manifest 0 文件（见 §4） |

> 注：`min=200` 护栏在 regulation/template 基本失效（中位数 75/31）。

## 2. 三类病理（附实样）

**P1. 标题/目录被切成独立 chunk（policy_national 主因）**
极碎样本全是裸标题或目录条目：`五、保障措施`(6字) / `二、重点研究方向`(8字) / `一、发展思路（题目）`(9字)。`十四五规划` 样本开头是 `目 录 TOC \o "1-2" \h \u HYPERLINK \l...`——Word 导出的 TOC 标记泄漏，2 个 chunk 含 TOC 噪声。6 字标题单独成块对 embedding 无信号，且与正文分离。

**P2. FAQ 问题与回答被拆开（kb_ops）**
样本：`1.账号登录，怎么办？ P2~3`(17字) / `2.项目统一...怎么查？ P4`(27字)——**只有问题，没有回答**。FAQ 应「问+答」一个 chunk，现在问题单成块，命中问题文本时拿不到答案。

**P3. 表单/表格字段标签碎片（regulation / template）**
样本：`4.研发投入情况`(11字) / `（一）研究背景和必要性 1`(12字) / `2.各直属单位研发投入汇总表（模板）`(22字)——制度文件里夹的表格/模板字段标签被逐个切碎。template 80 chunk 里 64% <50 字（spec 早就标注的「表单字段残留弱点」）。

## 3. 根因
`split_by_structure` 优先按**结构边界**（标题/列表项/表格行）切；`_merge_undersized` 的 min=200 护栏被「heading 不可跨越」规则架空——标题/字段标签这类低价值碎片并不回去。HANDOFF 的「碎片保留」决定本意是留住**有意义的**碎片（制度第X条、操作步骤），实际把**裸标题、FAQ 单问、表单标签**也留成了独立 chunk。

## 4. 副发现：kb_policy_group 当前 0 文件
`build_manifest_from_inventory` 在当前 `merged/merged/` 上给 kb_policy_group 产出 **0 个文件**（`集团规划` 目录的文件被 content-hash 去重到其他库的主 path）。`docs/kb_verify_report.md` 里「5 文件」是旧冻结态。**这是 manifest/去重问题，独立于 chunking**，需单独排查（见 §5 O-dedup）。

---

## 5. 所有提升方案（按类分组；收益/成本/风险）

### 清洗类（chunking 前去噪）
| ID | 方案 | 治什么 | 收益 | 成本 | 风险 |
|---|---|---|---|---|---|
| **O1** | 剥 Word TOC/目录噪声：正则去 `TOC \o` / `HYPERLINK \l` / `PAGEREF` / `目 录` 行及其下条目 | P1 | 高 | 低 | 低 |
| O2 | 表格识别保全：连续 `\|...\|` 行整体作一个 section，不按行切 | P3 | 中 | 中 | 低 |
| O3 | 复核 coreagent 残留清洗（`_strip_leading_meta_table` 等）是否漏清 | 通用 | 低 | 低 | 低 |

### 切块规则类（改 split_by_structure / guardrails）
| ID | 方案 | 治什么 | 收益 | 成本 | 风险 |
|---|---|---|---|---|---|
| **O4** | 标题并入后续正文：section=heading+body；禁 heading-only chunk（无后续 body 时并入前段） | P1 | 高 | 中 | 中（需保证不破坏有意义小节） |
| **O5** | 全局小-chunk 合并器（post-process）：切块后把 <N 字（如 80）chunk 并入相邻（不论结构边界） | 全部过碎 | 中高 | 低 | 低（N 可配） |
| O6 | 字符下限过滤：丢弃 <floor（20~30 字）纯噪声 chunk | P1/P3 噪声 | 中 | 低 | 中（怕误删有意义短句） |
| O7 | FAQ 问+答合并：识别 `数字.问题？` 模式，问题行与紧随回答合一个 chunk | P2 | 中 | 中 | 低 |
| O8 | 放宽「heading 不可跨越」：对 sub-floor（<min）的 heading section 允许并入前段，让 min 护栏真正生效 | P1 | 中 | 低 | 中（需调参） |

### 配置类
| ID | 方案 | 治什么 | 收益 | 成本 | 风险 |
|---|---|---|---|---|---|
| O9 | 按 KB 配 chunk 阈值：regulation/template 用更大 min（如 400）/ 更小 max | 过碎 | 中 | 低 | 低 |
| O10 | 按 KB 选 chunker 策略：template 走「整篇不切」或表单感知；FAQ 库走 Q-A 感知 | P2/P3 | 中高 | 中高 | 中 |

### 策略类（替换/补充切块算法）
| ID | 方案 | 治什么 | 收益 | 成本 | 风险 |
|---|---|---|---|---|---|
| O11 | 递归/语义切块（如 RecursiveCharacterTextSplitter）替换结构切分 | 通用 | 高 | 高（大改+重评） | 高 |
| O12 | 表单感知切块（template 专用） | P3 | 中 | 中 | 中 |

### 评测/护栏类
| ID | 方案 | 作用 | 收益 | 成本 | 风险 |
|---|---|---|---|---|---|
| **O13** | chunk 质量报告工具：入库后产出每库尺寸分布/碎片率报告（像 kb_verify_report） | 防回归 | 高 | 低 | 低 |
| **O14** | golden-query 召回评测集：典型问题对比改前/改后召回 | 数据驱动决策 | 决定性 | 中高 | 低 |
| O15 | 单测固化尺寸不变量：如「无 <20 字 chunk」「heading-only chunk=0」 | 防回归 | 中 | 低 | 低 |

### manifest/去重类（§4 副发现）
| ID | 方案 | 作用 |
|---|---|---|
| O-dedup | 排查 kb_policy_group 为何 0 文件：复核 `集团规划` 去重主 path 选择，必要时保留集团级文件主 path | 让 kb_policy_group 有可入库文件 |

---

## 6. 推荐子集（高性价比）
**先做（治标快、风险低）**：**O1（剥 TOC）+ O4（标题并入正文）+ O5（全局小-chunk 合并器）**——三者合力能消掉 policy_national 的 549 个 <30 字碎片，并普遍抬高 regulation/template 的中位数。
**配套护栏**：**O13（chunk 质量报告）**——任何改动前后跑一次，量化效果、防回归。
**决策依据**：**O14（golden-query 评测）**——真正回答「改了是否召回更好」；之前序列化阶段 deferred 的评测，这里正合适，建议在动 chunker 前先搭最小版。
**单独排查**：O-dedup（kb_policy_group 0 文件）。

> 任何 chunker 改动都会改变 `segment_id` → 需重灌受影响 KB（幂等：`MilvusWriter` 按 doc_id 删旧再插）。改前建议先 O13 出基线、O14 定评测口径。
