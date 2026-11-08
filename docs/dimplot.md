# UMAP 左下角短轴与 dimplot

`cs.pl.dimplot` 绘制图像，仅指定 filename 时保存，返回 Matplotlib Axes（多面板返回列表）。默认 `frameon=False`，每个子图左下角显示短箭头；`frameon=True` 保留常规坐标轴。

```python
cs.pl.dimplot(
    rna, reduction="umap", color=["sample_id", "leiden"],
    filename="umap", outdir=FIG, formats=("png", "pdf"),
    ncols=2, width=5, height=4, dpi=180,
    axis_length=0.1, axis_origin=(0.06, 0.06),
)
```

- `axis_length` 为面板宽/高的比例，默认 0.12，设为 0.1 表示约 10%；数字越小箭头越短。
- `axis_origin` 为面板相对坐标中的起点，默认 `(0.06, 0.06)`，表示距离左/下边界各约 6%。
- `dim1label` / `dim2label` 默认 UMAP1/UMAP2；绘制其他降维时可指定。
- 箭头使用 axes fraction，与数据坐标范围无关，不调整 UMAP 坐标或绘图区的数据限值。
- `width`/`height` 控制单面板尺寸，多面板保留 Scanpy 的布局，不压缩为一张单面板大小。
- `formats`、`outdir`、`filename` 控制直接保存；无需再调用 plt.savefig。调用后可以 plt.close("all")。

## Harmony 最大迭代数

现有 `cs.tl.harmony` 将额外参数传给 harmonypy，无需绕过 cellscope：

```python
fit = cs.tl.harmony(
    rna, batch_key="sample_id", use_rep="X_pca", key_added="X_pca_harmony",
    max_iter_harmony=50, random_state=0,
)
```

`max_iter_harmony` 是外层 Harmony 最大轮次；`max_iter_kmeans` 是每轮内部聚类的迭代上限，两者不同。`Stopped before convergence` 表示耗尽指定轮次但尚未满足数值收敛条件，不是 Python 异常。提高上限后可提前收敛；仍未收敛时需要检查目标函数及输入，不应只为消除提示放宽 epsilon_harmony。

重算 Harmony 会更新 PCA 校正表示；需要重做邻居图、UMAP、Leiden 等下游步骤才能使用新表示，只有重画图不会更新坐标。

## 修改记录

- 修复旧 dimplot 的箭头/标签写死为数据坐标的问题，增加 axis_length/axis_origin，保留原参数和直接保存行为。
- 使用局部 rc_context，避免永久修改全局 figsize；多面板不再压缩为单面板画布。新增返回值便于检查或继续编辑。
- 测试覆盖不同坐标平移与缩放下的数据限值不变、坐标未改写、多面板及正常坐标轴模式。
- 真实数据绘图及现有 Harmony 参数接口使用 sc 环境验证。测试中只改变图形显示；未改变分析模型或收敛阈值。
