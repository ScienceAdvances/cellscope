# 所有 cs.pl 函数统一的返回与保存方式

所有公开绘图函数都接受 `filename=None, outdir=".", formats=("pdf", "png"), dpi=300`。不指定 filename 时不保存、不创建输出目录，返回可继续编辑或保存的原生绘图对象。设置 filename 时一次保存多个格式，并仍返回该对象。`show=False` 默认不弹窗；show=True 在保存后显示。

```python
ax = cs.pl.qc(
    adata, keys=["total_counts"], groupby="sample_id",
    stripplot=False, inner="box", cut=0,
    figsize=(7, 4.5), xlabel="Sample", ylabel="Total UMI counts",
    title="Before QC", ylim=(0, None),
    filename="qc_before_total_counts", outdir=QC_FIG,
    formats=("pdf", "png"), dpi=300,
)
```

手动编辑并保存时不传 filename：

```python
ax = cs.pl.qc(adata, keys=["total_counts"], groupby="sample_id", stripplot=False)
ax.set_title("My QC plot")
ax.figure.savefig("my_qc.pdf", bbox_inches="tight")
# 也可用 plt.savefig；优先使用对象自己的 Figure，避免保存错误的当前图。
```

| 函数 | 返回对象 |
|---|---|
| qc、embedding、dimplot | 单图 Axes，多面板为 Axes 列表；后端选项也可返回 Figure/FacetGrid |
| pca_variance_ratio、volcano、cell_composition、cell_ratio | Axes |
| dotplot | 默认 Scanpy DotPlot，已生成图，可调用 get_axes()/savefig()；显式 return_fig=False 返回后端轴字典 |
| heatmap | Scanpy 的轴字典；如 `axes['heatmap_ax'].figure.savefig(...)` |
| plot_batch_effect | 图形对象字典：batch、cluster，以及需要计算 PCA 时的 pca_variance |
| phenotrack | 单患者/平均为 Axes，多患者为 Axes 列表 |
| phenotrack_ternary | Axes，投影坐标表在 cellscope_table |

不同绘图布局保留原生对象，不强制把多面板压成单个 Axes。所有输出都根据返回对象所属 Figure 保存，不使用 plt.gcf 猜测当前图。真正包含多张独立 Figure 的返回值会使用 filename_1、filename_2 等后缀。

`plot_batch_effect` 原本是计算及绘图的组合工作流，仍会按原方法更新图、UMAP 和聚类。未指定 filename 时不写图；指定 filename 时保存 filename_batch、filename_cluster 等。仅查看已有降维时使用 embedding/dimplot。

## 接口调整记录

- 移除旧 save 保存入口。cell_ratio 不再使用 od 保存，也不再返回表，而是返回 Axes；组成数据可从 `ax.cellscope_table` 获取。
- dimplot 的 filename 改为可选，默认不产生文件。
- 10 个公开 cs.pl 入口统一支持 filename/outdir/formats/dpi；新增统一导出模块，处理 Axes、Figure、列表、轴字典、DotPlot/FacetGrid，并在 show=True 时仍返回对象。
- qc 新增 figsize、xlabel、ylabel、title、ylim，允许自动保存前完成常用样式设置。
