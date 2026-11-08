# 一个合并 AnnData 的分批预处理接口

所有步骤默认使用 `adata.X`；只有显式指定 `layer` 时，QC/双细胞/阈值过滤才改用该层。CellBender 的 AnnData 接口固定校正 X，避免悄悄读取过期的 counts 层。

```python
# adata 已合并四个文库，obs 中保留 sample_id、原始 barcode，var 保留 gene_ids。
cs.pp.cellbender(
    adata, batch_key="sample_id", raw_files=raw_files, output_dir="models",
    expected_cells=expected_cells, total_droplets=25000, epochs=150,
)
adata = cs.pp.qc(
    adata, batch_key="sample_id", min_genes=200, min_cells=3, filter=True,
)
adata = cs.pp.scDblFinder(
    adata, batch_key="sample_id", remove_doublets=True, random_state=0,
)
adata = cs.pp.filter_qc(
    adata, batch_key="sample_id", method="mad", nmads=3, max_pct_mito=20,
)
```

## CellBender

`raw_files` 是 `{样本标签: 未过滤液滴H5路径}`。合并 adata 可以只装 Cell Ranger filtered GEX，不必把几十万空液滴全装入主对象；CellBender 内部按 batch_key 分组，从对应 raw 文件读取空液滴，依次拟合。

只有 filtered GEX 而没有空液滴数据，无法用这个接口凭空估计背景。不能将四份不同捕获文库混成一个 CellBender 模型。

- 校正前 X 完整备份到 `layers['raw_counts']`；已有此层时拒绝覆盖。
- 校正结果成为新的 X，`layers['counts']` 同步更新，防止后续误用旧 counts。
- 原细胞顺序、基因顺序、obs/var 元数据保持不变，不自动按 cell probability 丢细胞。
- 新增 `obs['cellbender_cell_probability']` 与 `obs['cellbender_is_cell']`。
- 按同批原始 barcode 与稳定 gene ID 对齐，并检查指定 raw 文件确实包含原 X 的观测计数。
- 所有批次成功后才提交矩阵；中途失败不把一个部分校正的矩阵写回 adata。
- 输出中若缺少目标 barcode，明确报错，需核对输入或提高 total_droplets，不静默补零。
- `expected_cells`、`total_droplets` 接受标量或样本字典。
- 默认不复用已有 H5。`reuse_outputs=True` 时，核对完成状态、原始文件 SHA256 及请求参数后才复用。
- 不归一化、不 log1p，不启动 CellBender 外部命令或 notebook。临时关闭的仅为 notebook 报告步骤。

原来的 `cs.pp.cellbender(raw_path, output_path, ...)` 文件接口仍兼容，也可显式使用 `cs.pp.cellbender_file`。

## 基础 QC

`min_genes` 指一个细胞至少检出多少基因；`min_cells` 指一个基因在一个样本内至少被多少细胞检出。最小阈值可传标量或完整的样本字典。逐批计算，并迭代到细胞/基因集合稳定。

合并矩阵必须共享一套基因轴，因此提供两种合并规则：

- `gene_join='union'`（默认）：基因在任一保留样本内达到 min_cells 即可保留；不修改其他样本中该基因的真实计数。
- `gene_join='intersection'`：要求每个仍有细胞的样本都达到 min_cells，可能丢失样本特有基因。

`varm['qc_gene_detected_cells_by_batch']` 与 `varm['qc_gene_pass_by_batch']` 记录逐样本支持情况，样本列顺序保存在 `uns['cellscope']['basic_qc']['params']['batch_order']`。不要把 union 理解为“每个样本都满足同一基因阈值”。

默认 `filter=False` 保留旧接口的只标记行为；设为 True 并接收返回值，才得到过滤后的对象。原始 X 备份层随行/列筛选同步切片；要恢复完整未筛选矩阵，读取 CellBender 后的检查点。

## scDblFinder

用户入口是 `cs.pp.scDblFinder`，默认用 X，后端为 scDblFinderPy，不经过 R。内部按文库独立运行，新增 `scDblFinder_score`、`scDblFinder_class`、`predicted_doublet`，并保存逐文库阈值。

`remove_doublets=False` 只注释；True 返回 singlet 子集。所有输入细胞的分数/标签保存在 `uns['scDblFinder_all_calls']`。不要将 doublet 预测理解为细胞类型或治疗响应结论。

## 高线粒体比例和极端 UMI/基因过滤

统一入口是 `cs.pp.filter_qc`：

- `method='mad'`：各样本内部计算 median ± nmads × MAD。默认对 log1p 总 UMI 与 log1p 检出基因数做双侧过滤；若将 pct_counts_mt 加入 mad_metrics，则该指标只做上侧过滤。
- `method='fixed'`：使用 min_counts/max_counts、min_genes/max_genes。
- `method='both'`：叠加以上两种规则。
- 三种模式均可通过 max_pct_mito 限制高线粒体比例细胞（20 表示 20%）。
- MAD 为原始 median absolute deviation（不乘正态一致性系数）。MAD=0 时默认跳过该指标的零宽阈值；`zero_mad='strict'` 则剔除偏离中位数的值，行为显式可选。
- 默认 `filter=True` 返回细胞子集；False 仅注释，便于检查。
- `remove_mito_genes=False` 默认保留 MT 基因。True 才删除基因，与剔除高 MT 比例细胞是两件事；指标基于删除 MT 基因之前的矩阵计算并保留。

失败原因写入 obs 的 `qc_fail_mito`、`qc_fail_mad`、`qc_fail_fixed`、`filter_qc_pass`。全部决策、逐样本阈值和摘要保存在 uns 中，可导出 CSV。

本接口属于用户指定的 Scanpy 分析扩展，不声称为 Rojas2023 原文的预处理步骤。CellBender 和 scDblFinderPy 使用已有官方 Python 后端，未重写科学算法。
