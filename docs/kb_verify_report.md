# KB 分类验证报告

## 总览
| KB | 文件数 | 类内凝聚度 |
|----|--------|------------|
| kb_ops | 16 | 0.8563 |
| kb_policy_group | 5 | 0.8754 |
| kb_policy_national | 164 | 0.7982 |
| kb_project | 1 | 1.0000 |
| kb_regulation | 12 | 0.8952 |
| kb_template | 2 | 0.9366 |

## 类间分离度矩阵（KB 质心两两余弦，越小越好）
| | kb_ops | kb_policy_group | kb_policy_national | kb_project | kb_regulation | kb_template |
|---|---|---|---|---|---|---|
| kb_ops | 1.00 | 0.80 | 0.75 | 0.55 | 0.82 | 0.70 |
| kb_policy_group | 0.80 | 1.00 | 0.88 | 0.57 | 0.93 | 0.79 |
| kb_policy_national | 0.75 | 0.88 | 1.00 | 0.56 | 0.82 | 0.74 |
| kb_project | 0.55 | 0.57 | 0.56 | 1.00 | 0.56 | 0.58 |
| kb_regulation | 0.82 | 0.93 | 0.82 | 0.56 | 1.00 | 0.82 |
| kb_template | 0.70 | 0.79 | 0.74 | 0.58 | 0.82 | 1.00 |

## 异常文件（margin < 0.05）
| 文件 | 当前归属 | 最近他库 | margin |
|------|----------|----------|--------|
| 0352fdaa55824280927f6632a62ffeb4_政策文件/广东省交通运输厅关于印发《广东省交通运输科技协同创新“.pdf.md | kb_policy_national | kb_policy_group | -0.0535 |
| 9ee6c2e017f5431ea0f1a63a3a95d93f_科研管理制度/1.科研管理制度_index.md.md | kb_regulation | kb_policy_national | -0.0326 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/广东省人民政府办公厅关于印发广东省人工智能赋能交通运输.pdf.md | kb_policy_national | kb_policy_group | -0.0309 |
| d9a64056d09f44b1a60f0788413389ae_政策_会议_讲话汇编/集团研发中心--政策汇编-20250512.xlsx.md | kb_policy_group | kb_policy_national | -0.0262 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/广东省综合交通运输体系“十四五”发展规划.pdf.md | kb_policy_national | kb_template | -0.0192 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2022-06-08_广东省交通运输厅关于加强交通基础设.md.md | kb_policy_national | kb_regulation | -0.0113 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/广东省人民政府办公厅关于印发广东省交通运输高质量发展三.pdf.md | kb_policy_national | kb_policy_group | -0.0080 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2017-11-24_《广东省交通运输科技项目结题工作规.md.md | kb_policy_national | kb_regulation | -0.0040 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-08-20_广东省交通运输厅关于印发《广东省交.md.md | kb_policy_national | kb_regulation | -0.0024 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/广东省科技创新“十四五”规划.pdf.md | kb_policy_national | kb_policy_group | -0.0018 |
| d9a64056d09f44b1a60f0788413389ae_政策_会议_讲话汇编/附件1 国家、部委科技创新政策汇编（2020年-2025年5月）-分类.docx.md | kb_policy_national | kb_policy_group | +0.0066 |
| d9a64056d09f44b1a60f0788413389ae_政策_会议_讲话汇编/附件3 “科技创新”相关会议重要论述汇编-2025.5.docx.md | kb_policy_national | kb_policy_group | +0.0125 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2016-11-15_广东省交通运输科技“十三五”发展规.md.md | kb_policy_national | kb_policy_group | +0.0134 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/数字交通“十四五”发展规划.docx.md | kb_policy_national | kb_policy_group | +0.0163 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2018-04-23_广东省交通运输厅关于征集2018年.md.md | kb_policy_national | kb_regulation | +0.0172 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2020-02-03_国家税务总局广东省税务局关于印发网.md.md | kb_policy_national | kb_policy_group | +0.0267 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-09-28_广东省市场监督管理局关于印发《广东.md.md | kb_policy_national | kb_regulation | +0.0278 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/高速公路收费_index.md.md | kb_policy_national | kb_template | +0.0288 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2014-11-06_广东省人民政府办公厅关于印发推进珠.md.md | kb_policy_national | kb_policy_group | +0.0302 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/粤交集基[2022]404号：关于印发《广东省交通集团科技创新“十四五”发展纲要》的通知.docx.md | kb_policy_group | kb_regulation | +0.0304 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/“十四五”交通领域科技创新规划.pdf.md | kb_policy_national | kb_policy_group | +0.0325 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2016-11-15_广东省人民政府关于印发《广东省系统推进全面创新改革试_200000177.md.md | kb_policy_national | kb_policy_group | +0.0350 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-01-22_《广东省科研诚信管理办法》（试行）.md.md | kb_policy_national | kb_regulation | +0.0360 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-01-16_关于印发《关于深入推进企业研发费用.md.md | kb_policy_national | kb_policy_group | +0.0368 |
| 60e0f7bac363446d9ba7b594f7bb9c2e_科小星-操作文档/广东省交通运输行业基建科技项目管理操作指引（交通集团体系内）.pdf.md | kb_ops | kb_regulation | +0.0368 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/数字交通“十四五”发展规划.pdf.md | kb_policy_national | kb_policy_group | +0.0385 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2015-07-23_广东省人民政府关于印发《广东省智能.md.md | kb_policy_national | kb_policy_group | +0.0394 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/交通运输部关于推进公路数字化转型加快智慧公路建设发展的意见-政府信息公开-交通运输部.pdf.md | kb_policy_national | kb_policy_group | +0.0399 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/“十四五”交通领域科技创新规划.doc.md | kb_policy_national | kb_policy_group | +0.0407 |
| 09690a19332d4cd68c28d9b44eb96f6f_操作指引_常见问题/广东省交通科技协同创新信息平台集团科技项目立项申报操作指引.pdf.md | kb_ops | kb_regulation | +0.0413 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-10-23_广东省人民政府关于印发广东省科技创.md.md | kb_policy_national | kb_policy_group | +0.0414 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/粤交集基〔2025〕285号：关于印发集团公路交通基础设施数字化转型升级科研选题指引的通知.pdf.md | kb_policy_group | kb_regulation | +0.0427 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-03-24_关于印发《广东省科学技术厅关于广东.md.md | kb_policy_national | kb_policy_group | +0.0436 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2020-05-06_广东省科学技术厅关于印发《广东省科.md.md | kb_policy_national | kb_policy_group | +0.0443 |
| 9ee6c2e017f5431ea0f1a63a3a95d93f_科研管理制度/广东省交通集团科技项目“揭榜挂帅”工作指引.docx.md | kb_regulation | kb_policy_group | +0.0444 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-09-04_广东省人民政府办公厅关于印发《广东.md.md | kb_policy_national | kb_policy_group | +0.0444 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/粤交集基[2022]404号：关于印发《广东省交通集团科技创新“十四五”发展纲要》的通知.pdf.md | kb_policy_group | kb_regulation | +0.0444 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2022-01-01_《广东省综合立体交通网规划纲要》_1483350327024553984.md.md | kb_policy_national | kb_regulation | +0.0447 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2015-10-10_广东省人民政府关于印发广东省深入实施知识产权战略推动_200000159.md.md | kb_policy_national | kb_policy_group | +0.0452 |
| 09690a19332d4cd68c28d9b44eb96f6f_操作指引_常见问题/（未整完）广东省交通科技协同创新信息平台集团科技项目立项申报操作指引.docx.md | kb_ops | kb_regulation | +0.0477 |
| 0352fdaa55824280927f6632a62ffeb4_政策文件/交通运输领域新型基础设施建设行动方案（2021—2025 年）.pdf.md | kb_policy_national | kb_policy_group | +0.0478 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-12-29_广东省交通运输厅关于印发《广东省普.md.md | kb_policy_national | kb_policy_group | +0.0485 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2021-12-31_广东省交通运输厅关于印发《广东省农.md.md | kb_policy_national | kb_policy_group | +0.0492 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/2019-07-03_广东省科学技术厅关于印发《广东省技.md.md | kb_policy_national | kb_policy_group | +0.0495 |
| 9e2bb1b888a5413e8ef6b3b7563cfed4_规划政策文件/创新平台类_index.md.md | kb_policy_national | kb_ops | +0.0500 |
