# cellscope：定位、架构与接口设计

状态：单细胞 RNA/VDJ 生态的基础版本已实现，2026 年 10 月。

本文是 [英文设计文档](DESIGN.md) 的中文版本，说明包的职责、数据约定、模块、接口、统计假设和扩展方向。接口名称与代码保持一致；“已实现”不等于已在所有数据规模和实验设计下完成验证。

## 1. 包的定位与职责边界

cellscope 面向单细胞基因表达和细胞状态分析，直接使用 scverse 生态的原生数据对象。它围绕 Scanpy、decoupler、PyDESeq2 和可选的 scvi-tools 模型，提供统一的工作流与接口。

cellscope 负责：

- RNA 质控、过滤、归一化和高变基因选择。
- 降维、邻居图、聚类和细胞类型/状态注释。
- 基因集评分、通路与转录因子活性分析。
- pseudobulk 聚合、样本层面的差异表达。
- 细胞组成统计、探索性比较和绘图。

immune 负责受体链、克隆型、免疫组库统计、纵向克隆追踪、克隆与细胞状态的关系，以及 bulk 与单细胞的匹配。cellscope 不依赖 immune；immune 只在可选的克隆相关表达分析中依赖 cellscope，避免循环依赖。

未来的 ATAC、空间组学等包可以采用相同的数据约定，但对应算法不属于当前版本。共享容器不意味着不同模态都适用 RNA 归一化，也不意味着空间 spot 与单细胞天然共享同一观察轴。

## 2. 通用数据约定

### 2.1 原生对象与结果存放

- 单模态函数接受 `AnnData`；多模态函数接受 `MuData`，默认通过 `mod="gex"` 选择 RNA 模态。
- `gex.X` 保存当前用于分析的表达矩阵。
- `gex.layers["counts"]` 保存原始、非负整数计数。
- `gex.obs` 保存 RNA 质控指标、聚类、细胞类型、细胞状态和评分。
- `gex.obsm` 保存表达降维结果，`gex.obsp` 保存图结构；后端结果保留在原生 `gex.uns` 位置。
- `uns["cellscope"][key]` 记录参数、后端名称及版本、数据约定版本；产生组成统计表的工具还保存结果表。
- 大型模型作为返回值交给调用者，不直接序列化进 `uns`，应使用模型自己的保存方法。

### 2.2 细胞身份与多模态对齐

细胞唯一标识采用 `library_id:barcode`，避免不同文库重复使用原始 10x barcode 时发生错误合并。

| 字段 | 含义 |
| --- | --- |
| `barcode` | 原始细胞 barcode |
| `library_id` | 成对 RNA/VDJ 数据使用的共同文库标识 |
| `sample_id` | 生物学样本标识 |
| `donor_id` | 生物学供体标识；一个供体可以有多个样本/文库 |
| `condition`、`time` | 实验条件、时间等研究元数据 |

模态特有的注释显式同步到全局 `obs`，名称采用 `gex:<column>`。样本、供体、文库、时间和条件等公共元数据，只有在对应细胞的值一致时才合并；冲突会报错。

默认外连接保留缺少某些模态的观察，并增加 `has_<modality>` 标记；内连接必须显式指定。质控先标记通过/不通过，不自动删除细胞。过滤 `MuData` 时同步对子模态进行子集选择。

会改变对象的分析拒绝 AnnData/MuData 视图；调用者应先执行 `.copy()`。持久化使用原生 `.h5ad` 和 `.h5mu`，不另造自定义容器，也不强制依赖 Muon 的分析模块。

## 3. 公共模块

```python
import cellscope as cs
```

| 模块 | 职责 |
| --- | --- |
| `cs.io` | 原生数据读取、细胞标识、多模态组装与保存 |
| `cs.pp` | 预处理与质控 |
| `cs.tl` | 表达/状态分析、样本层面的统计 |
| `cs.pl` | 绘图 |
| `cs.get` | 结果与注释提取 |
| `cs.datasets` | 可复现的示例数据 |

## 4. 主要接口

下列表格中的 `...` 表示省略了部分参数，不是可以直接执行的完整调用。

### 4.1 数据输入输出：`io`

| 接口 | 行为 |
| --- | --- |
| `io.read_10x(path, library_id=..., sample_id=..., donor_id=...)` | 调用 Scanpy 读取 10x HDF5/目录，并设置稳定的细胞标识 |
| `io.set_cell_ids(adata, library_id=..., ...)` | 在拼接样本或组装模态前设置细胞标识 |
| `io.concat(samples, sample_key="sample_id", join="outer")` | 拼接 RNA AnnData，拒绝重复细胞标识 |
| `io.create_mudata(modalities, join="outer")` | 组装任意模态，检查公共元数据一致性 |
| `io.read_h5ad`、`io.read_h5mu`、`io.write` | 原生对象读写；保存时检查文件扩展名 |

### 4.2 预处理：`pp`

| 接口 | 行为与输出 |
| --- | --- |
| `pp.qc(data, mod="gex", layer=None, min_genes=..., min_counts=..., max_pct_mito=...)` | 计算 Scanpy 质控指标及 `obs["qc_pass"]`，不直接删除细胞 |
| `pp.filter_cells(data, key="qc_pass", mod="gex")` | 返回显式过滤后的原生对象 |
| `pp.mad_outliers(data, metrics=..., batch_key=..., nmads=3)` | 按批次使用 MAD 标记稳健异常值 |
| `pp.normalize(data, layer=None, counts_layer="counts")` | 保存原始 counts，将归一化/log1p 结果写入 X；重复调用重新使用 counts |
| `pp.highly_variable_genes(data, batch_key=..., ...)` | 调用 Scanpy 选择高变基因，不删除其他基因 |
| `pp.neighbors(data, ...)` | 构建原生 Scanpy 邻居图 |
| `pp.doublets(data, layer="counts", batch_key=...)` | 在原始 counts 上运行 Scrublet，写入双细胞注释，不改变 RNA X |

### 4.3 分析工具：`tl`

| 接口 | 行为与返回值 |
| --- | --- |
| `tl.workflow(data, ...)` | 依次执行归一化、高变基因、PCA、邻居图、UMAP、Leiden，返回修改后的对象 |
| `tl.pca`、`tl.umap`、`tl.leiden` | 分别调用 Scanpy 对应步骤 |
| `tl.markers(data, groupby=..., method=...)` | 底层 Scanpy marker 排名，保留原生结果字段；默认 `use_raw=False` |
| `tl.find_markers(data, groupby=..., ident_1=..., ident_2=None, method=...)` | 指定细胞群与另一个细胞群/其余细胞比较，返回统一结果表 |
| `tl.find_all_markers(data, groupby=..., method=...)` | 每个实际存在的细胞群分别与其余细胞比较，支持检出比例与效应过滤 |
| `tl.annotate(data, mapping, reference_key=..., key_added="cell_type")` | 聚类标签映射到细胞类型，保留未注释细胞 |
| `tl.score_genes(data, gene_sets)` | 调用 Scanpy，基因集评分写入 obs |
| `tl.aucell(data, gene_list=..., network=...)` | 调用 decoupler 2.x AUCell，结果保存在 `obsm["score_aucell"]` |
| `tl.activity(data, network, method="ulm")` | 使用显式提供的通路/转录因子网络，生成原生 decoupler 评分 |
| `tl.scvi(data, layer="counts", batch_key=...)` | 返回 SCVI 模型，将潜在表示写入 obsm |
| `tl.scanvi(data, labels_key=..., unlabeled_category=...)` | 返回 SCANVI 模型，生成预测、概率和潜在表示 |
| `tl.reference_mapping(data, reference_model, ...)` | 通过 scvi-tools 适配查询数据，返回原生模型 |
| `tl.pseudobulk(data, sample_col=..., groups_col=..., metadata_cols=...)` | 调用 decoupler，按样本/分组求原始 counts 总和，返回 AnnData |
| `tl.differential_expression(pdata, method=..., design=..., contrast=..., groups_col=...)` | 分组运行 PyDESeq2、pylimma 的 TMM/voom 或 edgePython QL/LRT，返回统一结果表和原生模型 |
| `tl.pseudobulk_de(data, method=..., design=..., contrast=..., metadata_cols=...)` | 一步完成原始 counts 聚合与样本层面差异分析，返回结果表、聚合数据和模型 |
| `tl.de_methods()` | 查询可用方法、统计单位、后端与安装依赖组 |
| `tl.cell_composition(data, groupby=..., condition_col=..., donor_col=...)` | 生成完整的样本×类别计数与比例，保留零计数类别 |
| `tl.composition_test(table, condition_col=..., comparison=..., reference=..., donor_col=...)` | 样本层面的 Mann–Whitney 或供体配对 Wilcoxon 检验，跨类别进行 BH 校正 |
| `tl.trajectory(data, groupby=..., root=...)` | PAGA；指定根细胞后可计算 DPT |

### 4.4 绘图、结果提取与示例

`pl.embedding`、`pl.dotplot`、`pl.heatmap` 和 `pl.qc` 调用 Scanpy；`pl.cell_composition` 和 `pl.volcano` 使用 Matplotlib/Pandas。

新绘图接口返回原生坐标轴或绘图对象，不隐式保存文件。调用者可以传入坐标轴，并显式保存图像。

`get.obs_df`、`get.markers_df` 和 `get.result` 返回提取结果的副本。`datasets.toy_rna()` 生成可复现的原始计数示例。

## 5. 统计分析约定

### 5.1 细胞层面的 marker 分析

借鉴 Seurat 的 FindMarkers/FindAllMarkers 用法，提供统一的 `method` 参数，但不宣称与 Seurat 完全数值等价，也不照搬其默认过滤顺序和多重校正方式。

| `method` | 方法与用途 |
| --- | --- |
| `wilcoxon` / `wilcox` | Scanpy Wilcoxon 秩和检验 |
| `t-test` / `t` | Scanpy Welch t 检验 |
| `t-test_overestim_var` | 使用保守方差估计的 t 检验 |
| `logreg` | logistic regression 系数排名；不提供推断性 p 值，不等同于 Seurat 的 LR 似然比检验 |

输入为非负的 log1p 归一化表达矩阵，不使用原始 counts、缩放/整合后的表达或潜在空间。`use_raw=True` 指 AnnData 的 `.raw`，其中也应存放 log 归一化表达；名称为 raw 不意味着它必然是原始计数。

`ident_1`、`ident_2` 可为一个标签或标签列表，列表表示合并相应细胞；省略 `ident_2` 表示与其余细胞比较。`min_pct`、`min_diff_pct` 和 `features` 在检验前过滤基因；`logfc_threshold` 与 `only_pos` 在检验和校正后过滤结果。默认在每次比较内进行 BH 校正，也可选 Bonferroni。

统一结果包括 `gene`、`group`、`comparison`、`reference`、`log2FoldChange`、`pvalue`、`padj`、`method` 和 `unit`。marker 结果还包括评分与两侧的检出细胞比例。log2FC 基于恢复到线性表达后的平均值比值，正值表示 `ident_1` 更高。logistic regression 的 p 值字段保留缺失，不伪造显著性。

这些检验用于探索性 marker 发现，不对“细胞嵌套于供体”建模，不能把每个细胞当成独立生物学重复来证明条件效应。

### 5.2 pseudobulk 与差异表达

- 输入必须是非负整数 counts，不能用 log1p 或缩放后的表达矩阵代替。
- 样本协变量在同一样本内必须保持一致。
- 默认要求参与比较的每个条件至少有两个样本；这些样本应代表生物学重复。
- 配对研究显式使用类似 `~ donor_id + condition` 的设计。
- 技术重复文库应在推断之前合并。
- 细胞不是供体的独立生物学重复，不能用细胞数量代替样本数。

支持通过同一个 `method` 参数切换样本层面后端：

| `method` | 流程 |
| --- | --- |
| `pydeseq2` / `deseq2` | PyDESeq2 负二项模型与 Wald 检验 |
| `pylimma` / `limma` / `limma_voom` | edgePython TMM 归一化 + pylimma voom 精度权重 + 经验贝叶斯 moderated t 检验 |
| `edgepython` / `edgepython_ql` / `edger` | TMM + 负二项 GLM + quasi-likelihood F 检验 |
| `edgepython_lrt` | TMM + 负二项 GLM + 似然比检验 |

所有后端使用相同的 Patsy 设计矩阵与 contrast。`contrast=("condition", "post", "pre")` 的正效应表示 post 相对 pre。含交互项时，contrast 是在观察到的协变量分布上，分别将条件替换为两水平后设计行差值的平均，不自动等同于单个交互系数。

默认按总计数过滤并排除全零基因；可设置 `filter_method="filter_by_expr"`，复用 edgePython 的成熟表达过滤方法，为不同后端提供相同的受检基因集合。TMM/voom 在过滤后仍保留原始文库总量。零深度、缺失协变量、混杂导致的秩不足、无残差自由度等情况会报错。

多重校正在每个细胞分组的受检基因内进行，不默认跨整个研究的全部细胞类型与 contrast 校正。PyDESeq2 的独立过滤/Cook's 规则可能产生缺失 p 值或校正值，应保留而不是改成零。

`tl.differential_expression` 返回结果表和原生模型映射；`tl.pseudobulk_de` 还返回聚合后的 AnnData。`get.de_df` 可提取保存结果的副本。`uns["cellscope"][key]` 记录结果、方法、版本、公式、contrast、过滤设置、设计矩阵和对比权重，大型原生模型不写入 `uns`。

完整用法见 [差异分析指南](docs/differential-expression.md) 和 [可运行示例](examples/differential_expression.py)。不同 Python 移植后端的结果不保证与所有 R 版本一致；当前验证是适配器与实际 Python 后端的一致性，不是全包 R 数值等价认证。

### 5.3 细胞组成

组成比例以每个样本中已注释的细胞数为分母，保留零计数类别。

`composition_test` 是探索性的逐类别比例检验，不是完整的联合组成回归模型。配对数据应提供 `donor_col`；独立检验假设样本来自独立个体。低细胞数、采样不平衡和多组织研究需要针对实验设计建立模型。

## 6. 依赖与兼容性

基础依赖包括 AnnData、MuData、NumPy、Pandas、SciPy、Scanpy 和 Matplotlib。

| 可选依赖组 | 用途 |
| --- | --- |
| `clustering` | Leiden 聚类 |
| `functional` | decoupler 功能评分 |
| `differential` | pseudobulk 与 PyDESeq2 差异表达 |
| `limma` | John Mulvey 的 pylimma、voom 与 edgePython TMM |
| `edger` | edgePython QL/LRT 与表达过滤 |
| `de` | 安装全部差异分析后端 |
| `integration` | scvi-tools 模型与映射 |
| `dev` | 测试、格式检查和构建 |

重量级模型后端按需导入。导入 cellscope 不再需要无关的 `atopos` 工具包。

这里选择的 PyPI 分发包是 `pylimma`，不是 omicverse 的 `python-limma`；两者使用相同的 Python 导入名，不应混装，检测到冲突会报错。pylimma 与 edgePython 是 GPL 许可的可选依赖，本项目不复制其源码；分发软件或环境时应检查上游许可要求。

已有的 `pp.normalise`、`pp.fastqc`、`pp.mad_filter`、`tl.add_label`、`tl.find_all_markers`、`tl.deseq`、`pl.dimplot`、`pl.plot_batch_effect` 和 `pl.cell_ratio` 保留可访问性，新流程推荐使用上述新接口。

`tl.deseq` 现在要求明确的原始 counts 层，以及当前 PyDESeq2 所需的元数据和设计。包元数据与 `__version__` 保持一致。版本发布是独立操作，编写代码或通过测试不代表已经发布到 PyPI。

新的 `tl.find_all_markers` 默认返回统一字段，且默认 `use_raw=False`。需要旧版 `Feature` 索引、`Identy/LogFC/Padj` 字段时，可设置 `legacy=True`，或使用 `tools.gene_level_analysis` 中的兼容入口。

## 7. 成熟生态的复用方式

通过公共 API 调用成熟算法，不把第三方源码文件复制进项目。

| 项目 | 借鉴或复用内容 |
| --- | --- |
| [Scanpy](https://github.com/scverse/scanpy) | `pp/tl/pl/get` API 组织、图分析、marker 排名 |
| [AnnData](https://github.com/scverse/anndata) | 矩阵、注释存储和切片 |
| [MuData](https://github.com/scverse/mudata) | 多模态容器、元数据同步 |
| [decoupler](https://github.com/saezlab/decoupler-py) | 活性评分、pseudobulk |
| [PyDESeq2](https://github.com/owkin/PyDESeq2) | 基于计数的差异表达 |
| [pylimma](https://github.com/john-mulvey/pylimma) | voom、经验贝叶斯方差收缩 |
| [edgePython](https://github.com/pachterlab/edgepython) | TMM、QL/LRT、表达过滤 |
| [Seurat](https://satijalab.org/seurat/reference/findallmarkers) | marker 分析接口与参数组织思路 |
| [scvi-tools](https://github.com/scverse/scvi-tools) | 潜在模型、查询数据适配 |

以后如果复制具体源码，需要检查对应许可证并保留要求的声明。使用这些算法发表研究时，也应引用相应的方法论文。

## 8. 当前范围与后续路线

上述基础架构和接口已实现，RNA/VDJ 联合示例保存在 immune 中。可选深度模型适配器需要安装对应后端；接口存在不保证模型适合具体研究或能够收敛，初版没有完成实际深度模型训练验证。

后续方向包括参考图谱、模型基准测试、联合组成回归、邻域差异丰度，以及大数据和 backed 数组优化。空间 spot 映射与 ATAC 专用流程应由独立包负责，沿用通用数据约定，而不是全部塞入 cellscope。
