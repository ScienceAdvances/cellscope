# Python 原生环境 RNA 校正和双细胞检测

当前推荐使用合并 AnnData 的批处理接口：`pp.cellbender(adata, batch_key=..., raw_files=...)`、`pp.qc(adata, batch_key=..., filter=True)`、`pp.scDblFinder(adata, batch_key=...)` 和 `pp.filter_qc(...)`，默认以 X 为工作矩阵。详见 [一个合并 AnnData 的接口](adata_preprocessing.md)。以下文件级调用方式仍保持兼容。

## 接口与执行方式

- `cs.pp.cellbender(raw_file, output_file, ...)`：直接在当前进程调用 CellBender 0.4.0 的 Python 推断函数，返回 H5 路径。
- `cs.io.read_cellbender(output_file, raw_file=..., library_id=..., ...)`：读取官方输出，按 barcode 与稳定 gene ID 对齐原始计数。
- `cs.pp.scdblfinder_py(data, batch_key="library_id", ...)`：直接调用 scDblFinderPy，在不同捕获文库内分别运行。
- `cs.pp.scdblfinder(data, backend="python", ...)`：同一 Python 后端的统一入口。原有默认 `backend="r"` 保持兼容。

上述 Python 分支不调用 Rscript、shell 或其他分析脚本。CellBender 使用官方参数解析器获取默认配置，再直接调用推断函数；这不启动 CLI 进程。它的 notebook/HTML 报告回调被临时关闭，模型、后验、计数估计、PDF、metrics 和 checkpoint 逻辑保持官方实现。适配器同时屏蔽上游无条件输出的 HTML 报告成功消息，避免日志误报，并在 `.cellscope.json` 中记录 `upstream_notebook_report=false`。

## 安装与复现边界

CellBender 后端固定为 `0.4.0`。scDblFinderPy 官方仓库为 `ETHZ-INS/scDblFinderPy`，本次固定提交 `9f09ca0c4b491b517902130c7d42fb96427f970e`，版本 `0.1.0`，不是其他同名/相似名称的 Python 移植。

可选安装组为 `cellscope[cellbender]` 和 `cellscope[doublets-python]`。scDblFinderPy 要求 Python >=3.12，官方依赖版本固定得较严格。标准安装额外组会遵循这些约束。

本次在既有 micromamba `sc` 环境中保留 numpy/Scanpy 等核心版本，使用 `--no-deps` 安装未修改的 scDblFinderPy，并安装 CPU XGBoost 3.2.0。此环境与官方依赖锁定环境不同，已进行本地兼容性和直接调用对照，但不声称结果与官方锁定环境或 R scDblFinder 逐数值相同。分析项目中记录安装前后环境、源码哈希、安装日志及测试。

## 数据流

1. 每份文库的原始液滴（包含空液滴）分别运行 CellBender；不先取 TCR 子集。
2. 读取校正结果，按 `cell_probability > 0.5` 选择细胞。
3. RNA 基础 QC 后，在全部保留谱系中逐文库检测双细胞，再显式筛选 singlet。
4. 与同文库 TCR 连接，选择至少一条 productive TRA/TRB 且 CDR3 NT 有效的细胞。
5. 合并所选 RNA，重新进行标准化、HVG、PCA、聚类和注释。

双细胞预测不能先将 B/髓系等全部去掉，否则异型双细胞参照会减少。scDblFinderPy 本身尚无多样本分组功能，适配器循环文库，禁止把多文库静默混在一次预测中。所有输入矩阵保持原样，只添加分数/标签；细胞删除必须由调用者明确执行。

## counts 的含义

`read_cellbender` 返回：

| 位置 | 内容 |
|---|---|
| `layers['raw_counts']` | 观测到的整数 UMI，未去污染 |
| `layers['counts']` | CellBender 校正后的整数 UMI |
| `X` | 初始为校正 counts；`pp.normalize` 后为 log1p 标准化表达 |

适配器以 gene ID 对齐特征，校验校正计数不超过对应观测计数。gene symbol 重名时只修饰显示名称，不按重名符号错误对齐。还处理了官方读取函数在液滴数恰等于基因数时可能把特征元数据放入 `obs` 的情形。

CellBender 不会校正 TCR contig 序列；productive TCR 是独立的受体证据。仅有 TRA 的细胞可以进入 RNA 背景，但不能因此匹配到后续 bulk TRB 克隆。

## 最小示例

```python
import cellscope as cs

out = cs.pp.cellbender(
    "raw_feature_bc_matrix.h5", "results/cellbender.h5",
    expected_cells=6000, total_droplets=25000, epochs=150,
    fpr=0.01, use_cuda=False, n_threads=8,
)
rna = cs.io.read_cellbender(
    out, raw_file="raw_feature_bc_matrix.h5", library_id="P1Pre",
    sample_id="P1Pre", donor_id="P1",
)
cs.pp.qc(rna, layer="counts", min_genes=200, max_pct_mito=20)
rna = cs.pp.filter_cells(rna)
cs.pp.scdblfinder(rna, backend="python", layer="counts", batch_key="library_id")
rna = rna[rna.obs["scDblFinder_class"].eq("singlet")].copy()
```

expected_cells、total_droplets、QC 阈值均须根据数据检查，不是原论文给定的通用常数。正式去污染需要完整训练及模型诊断；两个 epoch 的接口测试不能用于生物学结论。本次新增步骤是用户选择的分析扩展，不冒称 Rojas2023 原文已经采用。

参考：[CellBender 官方代码](https://github.com/broadinstitute/CellBender)、[CellBender 使用说明](https://cellbender.readthedocs.io/en/latest/usage/index.html)、[scDblFinderPy 官方代码](https://github.com/ETHZ-INS/scDblFinderPy)。
