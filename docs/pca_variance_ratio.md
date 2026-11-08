# PCA 方差比例图

```python
ax = cs.pl.pca_variance_ratio(
    rna, n_pcs=50, log=True, tick_step=10,
    filename="pca_variance", outdir=FIG, formats=("pdf", "png"), dpi=300,
)
plt.close(ax.figure)
```

函数只读取 `uns['pca']['variance_ratio']`，不会重新计算 PCA、改变方差比例或表达矩阵。支持 AnnData 和 MuData（默认 mod="gex"）。

- X 轴标注 0、10、20、30、40、50 等数字，不逐点写 PC1–PC50。
- 数据实际从第 1 个主成分开始，0 只是坐标轴参考刻度。
- Y 轴为单个 PC 解释的方差百分比；log=True 是 Y 轴对数刻度，不会把数据改成 log 表达。
- 不会把已显示的 50 个 PC 的比例重新归一化到 100%。
- `n_pcs=None` 显示全部已计算 PC。若只计算了较少 PC，显示数量自动以已有数量为上限，并标出末尾实际 PC 数。
- `tick_step` 控制 X 轴刻度间隔；`color`、`figsize`、`title` 可修改外观；`ax` 可使用现有 Matplotlib Axes。
- 默认不保存、不弹窗；指定 `filename`（不含扩展名）、`outdir` 和 `formats=("pdf", "png")` 可一次输出多个格式，与 dimplot 的用法一致。
- `formats="png"` 也可只输出一种格式。只指定 filename 时默认输出 PDF 和 PNG。
- 只使用 filename/outdir/formats 保存；save 参数已移除，不提供兼容入口。`show=True` 可显示。返回 Axes。
- 如果存在 0 方差 PC，使用 log=False；0 不能用对数坐标显示。

新增独立绘图模块，不依赖 Scanpy 的逐 PC 文本标注。曲线、空心点、数字刻度和浅网格只改变显示，不影响下游 PC 选择或分析结果。
