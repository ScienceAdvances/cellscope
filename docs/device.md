# CPU / GPU 自动选择

```python
import cellscope as cs

cs.settings.device = "auto"  # 默认：按操作检查 GPU 和所需依赖，不能使用时提示并回退 CPU
cs.settings.verbosity = "info"
print(cs.device_info())
```

`cs.settings.device = "cpu"` 禁用下表接口的 GPU 路径。
`cs.settings.device = "gpu"` 严格要求这些接口使用 GPU；设备、依赖或参数不支持时抛出错误。
环境变量 `CELLSCOPE_DEVICE=auto|cpu|gpu` 可设置初始值。加载包不会初始化 CUDA。

## 单步覆盖

```python
cs.tl.pca(adata, n_comps=50, device="auto")
cs.pp.neighbors(adata, use_rep="X_pca", n_pcs=30, device="gpu")
cs.tl.umap(adata, random_state=0, device="gpu")
cs.tl.leiden(adata, resolution=0.6, device="gpu", flavor="cugraph")

# GPU 默认使用 Harmony1，可显式选择 Harmony2。
fit = cs.tl.harmony(adata, batch_key="sample_id", device="gpu", flavor="harmony1")
```

单步 `device` 优先于全局设置。旧参数 `use_gpu=True/False`、
`use_cuda=True/False` 仍有效，分别等价于严格 `gpu` 和 `cpu`；
不传这些旧参数时才继承全局设置。旧参数与单步 device 冲突会报错。
因此要使用自动模式，应删掉原代码中的 `use_gpu=True` 或 `use_cuda=True`，
改成 `device="auto"`，或不指定设备参数。

## 已接入的接口

| cellscope 接口 | GPU 后端 | CPU 后端 |
|---|---|---|
| pp.normalize | RAPIDS normalize_total + log1p | Scanpy |
| pp.highly_variable_genes | RAPIDS HVG | Scanpy |
| pp.neighbors | RAPIDS/cuVS | Scanpy |
| tl.pca | RAPIDS/cuML | Scanpy |
| tl.umap | RAPIDS/cuML | Scanpy |
| tl.harmony | RAPIDS Harmony1/Harmony2 | harmonypy 原生版本 |
| tl.leiden | RAPIDS/cuGraph | Scanpy |
| tl.markers | RAPIDS rank_genes_groups | Scanpy |
| tl.score_genes | RAPIDS score_genes | Scanpy |
| tl.workflow | 逐步骤选择上述后端 | Scanpy |
| pp.scDblFinder / pp.scdblfinder_py | scDblFinderPy + RAPIDS + CUDA XGBoost | scDblFinderPy + Scanpy |
| pp.scdblfinder(backend="python") | 同上 | 同上 |
| pp.cellbender / pp.cellbender_file | PyTorch CUDA | PyTorch CPU |

RAPIDS、scDblFinderPy、PyTorch 分别检测。PyTorch 能使用 GPU，不代表
CuPy、cuGraph 或 XGBoost 也能使用。检测包含一次小规模 CUDA 运算，
并检查相关库是否可导入；不保证后续任意规模数据都不会发生显存不足。
结果在进程中缓存，修复依赖或环境后可调用 `cs.device_info(refresh=True)` 重新检测。

GPU 接口使用临时副本，只搬运所选 assay，避免将所有 counts/raw_counts layer
一起占用显存。PCA、HVG、近邻、分群等步骤不改变原始 X 或计数 layer；
normalize 按原有接口写回归一化 X 并保留 counts。返回对象中的矩阵为 CPU 类型，
可直接供 Scanpy、immune 使用并写入 h5ad。临时副本会增加 CPU 内存和传输开销。

## 参数、算法和结果

显式的 CPU 算法选择会被保留。例如 `flavor="igraph"`、
PCA 的 `svd_solver="arpack"`，以及 RAPIDS 不支持的参数，
在 auto 下提示并使用 CPU，在 gpu 下报错。要使用 GPU Leiden，
可使用 `flavor="auto"`（默认）或显式 `flavor="cugraph"`；
不要同时指定 CPU 的 `flavor="igraph"` 或 `flavor="leidenalg"`。
显式 cugraph 会严格要求 GPU，不会回退到其他算法。
GPU Leiden 建议使用 `n_iterations=100`（RAPIDS 默认值），不要直接沿用
CPU igraph 常见的 `n_iterations=2`。真实样本同图对照显示两次 GPU 迭代
可能产生上千个小群，而 10/100 次得到相同的九群结果。
内置 workflow 已设置 GPU=100、CPU igraph=2；显式 GPU 迭代次数不超过 2 时会提示警告。
现有 `tl.workflow` 在 CPU 回退时保留 igraph，在 GPU 时选择 cuGraph。

GPU 与 CPU 实现不保证数值、近邻、UMAP 坐标、Leiden 标签或 marker p 值完全一致。
切换后端后应重新计算依赖它的后续步骤，不应将新旧图与标签混用。
每步后端见 `adata.uns["cellscope"]["devices"]`，普通日志也会显示。

自动回退仅针对预先检测到的设备/依赖不可用、或明确不支持的参数。
运行中的算法错误、显存不足及 CUDA 内核错误会保留异常，不会吞掉后重跑 CPU。
GPU 临时结果在操作成功前不写回原对象。

## 其余功能

这不是将整个包的每个函数改写为 CUDA。读取文件、绘图、QC 阈值筛选、
R 方法、统计模型和部分第三方模型仍使用各自原来的实现。
`tl.harmony` 现在支持 GPU：device 遵循统一设置。CPU 保留原生 harmonypy
默认参数和拟合对象；GPU 未指定 flavor 时明确使用 Harmony1，可显式设置
`flavor="harmony1"` 或 `"harmony2"`。显式 flavor 仅适用于 GPU；若 GPU
不可用会报错，不会把指定的算法悄悄替换成 CPU 原生默认算法。

注意：当前机器 harmonypy 2.1.0 的默认 lamb=None 使用动态正则化，
不能把 CPU 原生默认设置简单称为原始 Harmony1。CPU 记录 flavor 为
harmonypy_native，GPU 记录实际 flavor，不声称两者方法或数值完全相同。

GPU 返回 GPUHarmonyResult，包含 cells-by-PCs 的 Z_corr、backend 和 flavor，
没有原生拟合对象的迭代历史等字段。CPU 返回原生 fit 保持兼容。
两者的校正坐标均写入指定的 adata.obsm[key_added]。GPU 只复制 PCA 和批次元数据，
不复制或搬运整个表达矩阵。

常用 harmonypy 参数会映射到 RAPIDS：nclust→n_clusters、lamb→ridge_lambda、
block_size→block_proportion、max_iter_kmeans→max_iter_clustering、
epsilon_cluster→tol_clustering、epsilon_harmony→tol_harmony。
默认未显式传入的其他参数仍采用各自后端默认值，不承诺默认参数完全相同。
重复指定映射前后同一参数会报错。CPU 专有参数（例如 ncores）在 auto 模式下
选择 CPU，在严格 gpu 模式下明确报不支持，不会被忽略。
GPU 的 lamb=None 或向量 lambda 不会自动改写成固定标量。
scVI/scANVI 等模型仍通过它们自己的 train_kwargs 配置设备。

消息遵循已有 `cs.settings.verbosity`；`quiet` 关闭 cellscope 日志，
`warning` 只显示回退等警告。第三方库原生警告仍可能显示。

## 本项目机器的 CUDA 路径

验证时发现 CUDA 头文件位于：
`/home/tim/mamba/envs/sc/targets/x86_64-linux/include/cuda_runtime.h`。
PyTorch 可直接用 GPU，但 CuPy 在未指定路径时找不到头文件。
启动分析前可以在终端设置：

```bash
export CUDA_PATH=/home/tim/mamba/envs/sc/targets/x86_64-linux
```

这个路径仅适用于当前机器。包不会自动安装包、修改系统驱动或写入 shell 配置。
