# PhenoTrack：用已有单细胞状态追踪克隆表型

本模块接收 Scanpy 分群/人工注释或单细胞状态概率，不使用 GeneVector。
输入的 clone_key 由 immune 或用户上游流程定义；本模块不会重新定义 TCR 克隆，
也不执行 CloneTrack 的疫苗诱导克隆筛选。

## 基本使用

```python
import cellscope as cs

result = cs.tl.phenotrack(
    rna,
    clone_key="clone_id",        # 也可用已经回填的 track_clone_id
    phenotype_key="cell_type",   # Scanpy + marker 审阅后的标签；功能测试可用 leiden
    donor_key="donor_id",
    sample_key="sample_id",
    time_key="condition",
    time_order=["pre", "post"],  # 显式指定真实顺序，不自动按名称排序
)

cs.pl.phenotrack(
    result, filename="phenotrack_by_donor", outdir="figures",
    formats=("pdf", "png"), dpi=300,
)
```

支持 AnnData 和 MuData（默认读取 gex 模态）；上述列需在所选模态 obs 中。
每个输入 barcode 只能对应一个已解决的克隆身份。多候选匹配不能随意选一个，
应在上游明确处理。不同患者即使 clone_id 字符串相同，也始终分开计算。
同一 sample_id 只能对应一个患者和一个时间；同患者同时间多个 sample 会按细胞合并，
不是将技术文库当作生物学重复。

函数不访问、归一化、修改 X 或 layers，只读取 obs 和可选的 obsm 概率。
统计采用 NumPy/Pandas CPU 汇总，不需要新增 GPU、R 或 PhenoTrack 依赖。
现有日志等级和批量患者进度适用。

## 作者方法与统计口径

默认 denominator="all_cells"：分母是输入对象中每位患者、每个时间点的全部细胞数，
与作者 cell_list 长度的归一化对应。输入应是你希望作为组库分母的细胞全集，
例如已完成 QC 的 T 细胞。选择目标克隆时要使用 clones 参数，
不要先对子集化对象后仍将输出解释为原全集频率。

```python
result = cs.tl.phenotrack(
    rna, time_key="condition", time_order=["pre", "post"],
    clone_key="track_clone_id", phenotype_key="t_state",
    clones={"P1": ["target_1", "target_2"], "P8": ["target_3"]},
)
```

clones=None 使用所有有效克隆；也接受跨患者通用的克隆列表。
缺少 clone 或状态的细胞不贡献表型质量，但默认仍保留在分母。
clone_definition 可填写上游身份的文字说明并保存到参数；它只记录定义，不改写克隆 ID。
denominator="annotated_cells" 是显式替代口径，仅以同时有克隆和状态的细胞为分母，
会记录到参数，不能与 all_cells 频率直接混用。

对克隆 c，记其在时间 t 的状态比例为 q_i(c,t)，克隆频率为 f(c,t)：

```
outflow(c,i→j) = q_i(c,t) × q_j(c,t+1) × f(c,t)
inflow(c,i→j)  = q_i(c,t) × q_j(c,t+1) × f(c,t+1)
```

因此带宽两端可以不相等，反映克隆丰度变化。静态柱包括选中克隆中只在一个时间检出者，
但只有两端都检出的克隆有跨时间流带。没有采样的患者/时间不会当作零丰度或消失。
normalize=False 对应作者细胞数模式，把上述 f 替换为克隆表型细胞质量。

图中不是同一个物理细胞的移动或谱系追踪；appeared/disappeared 仅表示采样检出情况，
不等于真实产生/死亡。细胞状态和克隆身份可以按用户需求替换，但不能混用不同定义的 ID。

## 状态概率

硬标签相当于每个细胞一个 one-hot 向量。也可提供 obsm 中的概率 DataFrame：

```python
result = cs.tl.phenotrack(
    rna, time_key="condition", time_order=["pre", "post"],
    probabilities_key="t_state_probabilities",
)
```

概率 DataFrame 的行必须与 obs_names 对齐，列名为状态名；若输入普通二维数组，
必须同时提供 phenotype_order。每行必须在 [0,1] 且和为 1，或全零表示未分配。
不会默默归一化错误概率。输出 cell_mass 可为小数，表示概率之和，而非真实细胞数增加。

## 结果与保存

result 是 cs.tl.PhenoTrackResult，包含：

| 属性 | 内容 |
|---|---|
| composition | 患者 × 时间 × 克隆 × 状态的细胞质量、克隆内比例和样本频率 |
| nodes | 所有选中克隆的静态状态柱高，含 sampled 标记 |
| flows | 共享克隆各状态对的 outflow / inflow |
| flux | 克隆内状态比例的 L1 变化，范围 0–2 |
| events | 各相邻采样时间的 shared / appeared / disappeared |
| denominators | 筛选目标克隆前的细胞数、可注释细胞数、实际分母 |
| parameters | 输入列、顺序、克隆选择、分母、归一化、方法和作者源文件 SHA256 |

flux 的 min_cell_count 默认 0，按作者严格 > 阈值同时检查两端，
仅作用于 L1 统计，不会偷偷过滤静态柱或流带。L1 不是 P 值或显著扩张检验。

结果也写入所选模态的 uns['phenotrack']（可用 key_added 更名），
支持通过 cs.io.write 保存到 h5ad/h5mu，并通过 cs.pl.phenotrack(adata) 直接读取绘图。

```python
result.write("phenotrack_tables")  # CSV + parameters.json，新目录或空目录
result.write("phenotrack_tables", overwrite=True)  # 显式允许更新这些表
```

## 多患者平均

```python
mean = cs.tl.phenotrack_aggregate(result)
cs.pl.phenotrack(mean, filename="phenotrack_mean", outdir="figures")
# 或 cs.pl.phenotrack(result, aggregate=True, ...)
```

默认等患者权重，不按测到的细胞数或克隆数加权，与作者完整队列场景对应。
默认 missing="error" 要求每个患者在全部时间均有可用分母。
missing="available" 是显式的可变队列替代方案：节点各自按本时间可用患者数平均，
流带两端按各自时间的患者数缩放，只有有相邻两次采样的患者贡献连线。
参数中记录每个时间点的患者数，避免将未采样当成克隆阴性。
如果某一轴位置所有患者都没有单细胞数据，available 模式保留 NaN 并标为 not sampled，
不会绘制零丰度柱或插值填充状态。normalize=False 下的等患者细胞质量平均是显式扩展；
与作者多人绘图代码的数值对照使用频率模式。
不复制作者在不完整队列图中可能遗漏仅终点存在患者节点的行为。

## 不平衡流图

cs.pl.phenotrack 默认每位患者一面板，支持 donor="P1" 选择一个患者。
状态顺序默认沿用 categorical 顺序；也可在分析时指定 phenotype_order。
phenotype_colors 提供状态颜色映射；time_positions 可使用真实天数等递增位置。
默认 interpolation="hill" 使用作者同形 Hill 曲线及 1000 个插值点；
interpolation="linear" 是显式可选的显示替代方案。
state_gap 默认 0，可增加状态柱间隙；不会改变统计值。
显示参数保存在返回 Axes 的 cellscope_parameters。

与所有 cellscope 绘图一致：filename/outdir/formats/dpi 控制保存。
不传 filename 不产生文件，返回可编辑 Axes；多患者返回 Axes 列表。
不支持旧 save 参数。

## 三状态三角图

```python
ax = cs.pl.phenotrack_ternary(
    result, donor="P1", from_time="pre", to_time="post",
    phenotypes=["Cycling", "Effector", "Memory"],
    filename="clone_ternary", outdir="figures", formats=("pdf", "png"),
)
```

使用作者投影，三个状态分别在上、左、右三个顶点。单时间点可省略 to_time；
双时间点画状态组成变化的箭头（也支持 line_type="lines"）。
默认 conditional=True 在选择的三个状态内重新归一化，与作者投影逻辑一致。
多于三个状态时必须显式选择，避免无提示丢弃其他状态；conditional=False 会拒绝剩余状态质量。
默认考虑起点检出克隆；include_new=True 额外显示终点才检出的克隆，属明确的显示扩展。
多患者时必须选择 donor，不支持将不同患者的克隆混成一条轨迹。
坐标表在 ax.cellscope_table，投影选项在 ax.cellscope_parameters。

## 原函数对应与验证

| 作者接口 | cellscope 接口 |
|---|---|
| plot_pheno_sankey | cs.tl.phenotrack + cs.pl.phenotrack |
| plot_agg_pheno_sankey_w_dict | cs.tl.phenotrack_aggregate / aggregate=True |
| get_phenotypic_flux | result.flux，L1 与相同严格计数阈值 |
| ternary_plot_projection / plot_ternary_phenos | cs.pl.phenotrack_ternary |

对外部未修改作者源码进行了静态柱高、两端流量、L1、完整队列等患者平均和投影的数值对照。
测试配置 PHENOTRACK_REFERENCE_DIR 与 CLONETRACK_REFERENCE_DIR 后启用；
参考源码不随包分发。作者旧 SciPy 导入名仅在测试中做兼容别名，不改变计算。
独立实现修复了作者单患者图不传 times 时 dx 未定义的问题，默认位置可直接使用。
不承诺像素级复刻原图，Scanpy 标签也不要求等同于作者的表型标签。

参考：https://github.com/zsethna/PhenoTrack
