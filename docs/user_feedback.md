# 运行消息与 verbosity 等级

默认使用 `info`，终端只显示当前样本、关键统计和主要操作的开始/完成。内部阶段、参数、每轮训练和定时状态提示只在 `debug` 下显示。分析参数、计算和返回值不受消息等级影响。

## 设置等级

```python
import cellscope as cs

cs.settings.verbosity = "info"     # 默认：简洁进度、当前样本、关键结果
cs.settings.verbosity = "debug"    # 排查问题：参数、内部步骤、所有训练轮次、定时状态
cs.settings.verbosity = "warning"  # 只打印 WARNING 消息
cs.settings.verbosity = "quiet"    # 关闭 cellscope 消息
```

也可以用 `cs.logging.configure(verbosity="debug")`。切换立即生效，无需重新导入。`error` 等级保留兼容，仅显示错误；整数兼容映射为 0=error、1=warning、2=info、3=debug。

`warning` 按只显示警告处理，不打印 INFO、DEBUG 或 ERROR 日志；`quiet` 不打印任何 cellscope 日志。这两个模式都不会吞掉实际 Python 异常，失败仍会正常抛出。

## info 与 debug 的区别

以下只是格式示例，不代表真实分析结果：

```text
[10:00:00] [cellscope] INFO START pp.scDblFinder | 26,504 cells x 38,606 genes
[10:00:00] [cellscope] INFO START scDblFinderPy [1/4] P1Pre | 6,238 cells
[10:00:20] [cellscope] INFO scDblFinderPy P1Pre | singlets=5,800 | doublets=438 | threshold=0.6000
[10:00:20] [cellscope] INFO DONE scDblFinderPy [1/4] P1Pre | elapsed 20.0s
...
[10:02:00] [cellscope] INFO DONE pp.scDblFinder | ... | elapsed 120.0s
```

- `info`：主要函数开始和完成、每个样本的处理、保留细胞数、双细胞数量等关键统计。不显示嵌套函数的开始/完成、完整参数、内部阶段、CellBender epoch 或周期性 RUNNING 消息。
- `debug`：在上述信息之外，显示完整诊断参数、内部函数调用、QC 每次迭代、原始文件校验/对齐、CellBender 每个 epoch 和当前阶段。长任务默认每 30 秒输出一次 RUNNING，结束或失败后停止。
- scDblFinderPy 默认 `verbose` 仅在 `debug` 下启用。调用者显式传入 `verbose=True` 时优先尊重该参数。
- CellBender 后端 INFO 消息映射为 cellscope DEBUG，后端警告和错误保留原等级；其原生完整文件日志不被删减。
- INFO 下的长任务在样本开始与完成之间可能较长时间没有新行；需要观察训练细节时可随时切换到 DEBUG。

```python
cs.settings.heartbeat_interval = 30  # DEBUG 定时提示间隔；0 关闭
```

## 启动 Logo 与环境变量

INFO/DEBUG 导入时显示一次 ASCII Logo 和版本；WARNING/ERROR/QUIET 不显示 Logo。需要在导入前控制等级：

```python
import os
os.environ["CELLSCOPE_VERBOSITY"] = "quiet"  # 或 info/debug/warning
import cellscope as cs
```

`CELLSCOPE_BANNER=0` 只隐藏 Logo；`CELLSCOPE_QUIET=1` 同时关闭 Logo 和消息；`CELLSCOPE_HEARTBEAT_SECONDS` 设置 DEBUG 状态提示间隔。同一进程重复 import/reload 不反复显示 Logo。

## 日志文件及第三方输出

```python
cs.logging.configure(verbosity="debug", log_file="analysis.log")
cs.logging.configure(log_file=None)  # 停止写 cellscope 日志文件
```

cellscope 日志文件跟随同一个 verbosity 等级。默认控制台写到 stderr 并立即 flush，不污染 stdout 中的 CSV/JSON。重复配置不叠加 handler，不修改 Python root logger。

此设置控制 cellscope 消息及接入的后端消息，不会全局屏蔽 Python `warnings.warn`、异常回溯或第三方自行配置的输出。用户显式开启的第三方 verbose 也由对应参数控制。

## 本次调整

- 将详细阶段、嵌套操作、训练轮次和定时状态提示降到 DEBUG；INFO 保留样本进度和关键结果。
- 基础 QC 每个样本只在首次迭代提示处理，详细迭代留在 DEBUG；最终逐样本保留数仍显示。
- 默认 scDblFinderPy verbose 跟随 DEBUG；WARNING 只显示警告，QUIET 关闭消息，低输出等级不显示启动 Logo。
- 新增等级矩阵、嵌套步骤、后端日志、Logo 和 INFO 不启用定时线程的测试。此次只修改 cellscope，未修改 immune 或数据分析脚本。
- 本次验证：全套测试 102 passed、29 skipped；实际安装目录相关测试 30 passed；Ruff 和 git diff --check 通过。

更新包文件后，请重启现有 Python 进程，以加载修改后的消息实现。此后切换等级可直接修改 settings。

## 初版实现与验证记录（历史）

- 新增统一消息模块与公开 `cellscope.logging`，导出 `cs.settings`；不增加第三方依赖。
- I/O、预处理、分析、绘图的已加载公开函数统一报告开始/完成/失败，保留原签名和返回对象；旧版惰性加载入口同样提供消息。
- 补充逐文库 CellBender/scDblFinderPy、逐批 QC、逐群 marker、逐基因集评分，以及归一化/HVG/PCA/Leiden 等阶段消息。I/O 用 barcodes/features 描述形状，避免把原始空液滴误称为细胞。
- 修改仅限 cellscope 包及其 README、测试和文档，没有修改分析项目的教学脚本或数据，也没有改动 immune。
- 完整测试：94 passed、29 skipped。最终相关功能和反馈测试：36 passed、5 skipped；跳过项为未安装的相应可选依赖。
- 额外运行 CellBender 官方 CPU 后端的 80 个模拟细胞、2 epochs 验证：样本编号、原生 epoch、heartbeat 和完成信息实时可见，raw_counts 备份逐元素一致，后端原有日志 handler 在结束后恢复；未启动外部分析进程。此项仅验证消息接入，不是新的真实样本生物学分析。
- 消息测试覆盖启动 Logo 开关、重复 import/reload 不重复输出、stderr/stdout 分离、静默设置、文件 handler 不重复、实时 heartbeat、成功/失败后线程退出、原异常原样抛出、数据不变和函数签名保持。

更新已有安装后，请重新启动 Python 进程再导入 cellscope，以加载所有新增的消息逻辑。
