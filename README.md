# FYP · 器件约束联合Gaussian与视觉动作识别

## 综合实验报告

完整报告：[器件约束视觉动作识别实验报告](docs/器件约束视觉动作识别_实验报告.md)。以KTH为主体，按“选择原因→研究方向→实验流程→对应结果”组织；HMDB51仅简述为前期效果不理想、因此转用KTH的背景，不展开数据或成绩。KTH各轮仍区分模型与Cycle协议，保留负面结果及探索性限制。

精确数据：[27行KTH跨阶段汇总CSV](docs/assets/experiment_summary.csv)与[带源文件哈希的JSON](docs/assets/experiment_summary.json)。运行`python scripts/build_experiment_report_summary.py`可从已有结果重建汇总，不重新训练或评价测试。

GitHub仅发布代码、报告、PNG、汇总指标、样本划分及检查记录。原视频、环境、二进制特征／响应缓存、模型和大型KTH逐样本得分CSV保留本地；历史文档指向这些本地文件的链接不属于仓库交付。器件实验还依赖独立原项目及实测响应库，新机器需提供这些依赖并调整路径，不能仅克隆仓库直接重放所有实验。本文后续各轮记录和命令保留原样。

## 最新实验 固定时空输入的Gaussian有限调参

固定KTH新16维输入、256个训练中心、响应库与原联合Gaussian计算，已完成72个Gaussian、4个Linear及12个理想软件参考候选。参数只按训练／验证选择，测试前锁定；未下载、重采样、修改物理曲线或操作硬件。旧默认模型、预测和历史结果均保留。

| 方案 | 调整前测试Cycle准确率 | 调整后 | 调整后Macro-F1 |
|---|---:|---:|---:|
| Linear | 50.58% | 50.81% | 0.4763 |
| G0 | 53.72% ± 0.69% | 53.56% ± 1.01% | 0.5293 |
| G1，验证预选 | 53.60% ± 0.59% | **54.01% ± 0.62%** | **0.5322** |

±为同一862片段10轮响应抽样标准差。验证选择sigma=0.25、校准alpha=0.001、头锚定alpha=0.01；G1实际6080行头alpha=0.08。Linear头alpha=10。G1仅提高0.42个百分点，跨人4折未改善，不能宣称稳定优势；验证变化主要来自头部正则化，而非改变宽度。拳击→拍手错误率降至52.52%，但其他类别有退化，暂不直接进入完整硬件实验。

```bash
cd /home/ziyan/FYP_XZH
/home/ziyan/FYP_2026/.venv/bin/python -m unittest test_tune_kth_spatiotemporal -v
/home/ziyan/FYP_2026/.venv/bin/python -u tune_kth_spatiotemporal.py
/home/ziyan/FYP_2026/.venv/bin/python tune_kth_spatiotemporal.py --verify-only
/home/ziyan/FYP_2026/.venv/bin/python render_kth_spatiotemporal_tuning.py
```

同条件完成后整轮直接复用，不重复搜索。25项测试及最终模型重载通过；主流程20.20秒、CPU单线程、峰值RSS约727.6 MiB，无GPU。结果与源码保存在`outputs/kth_spatiotemporal_tuning_v1/`，详见[中文报告](docs/KTH_固定时空输入Gaussian有限调参结果.md)、[结果表CSV](outputs/kth_spatiotemporal_tuning_v1/results_table.csv)、[结果表PNG](outputs/kth_spatiotemporal_tuning_v1/results_table.png)、[四联图](outputs/kth_spatiotemporal_tuning_v1/four_panel_summary.png)。本轮仍为已查看测试集上的探索性筛选和处理后响应库索引留出，不是独立硬件验证。

## 上一轮实验 16维时空输入对照

已完成同一KTH数据和器件模型下的输入对照：原16个全图时间帧差，对比“4个固定画面区域×4个时间组”的16维运动量。采样位置和基础灰度处理不变；重算2389段旧特征最大差0，旧模型及成绩直接复用。新输入的缩放、256中心、标准化、校准及分类头只由训练部分重新拟合，没有覆盖旧缓存、下载视频或操作硬件。

| 方案 | 旧输入Cycle准确率 | 新输入Cycle准确率 | 新输入Macro-F1均值 |
|---|---:|---:|---:|
| Linear16 | 44.90% | 50.58% | 0.4758 |
| G0 | 49.34% ± 0.57% | 53.72% ± 0.69% | 0.5233 |
| G1，验证预选 | 50.75% ± 1.04% | **53.60% ± 0.59%** | **0.5209** |

目标sigma=0.25、因子裁剪[1e-6,1]、Cycle索引240／80／80、每段8个训练Cycle及等价Ridge正则化规则均保持不变。±为同一片段10轮响应抽样标准差。新G1平均提高2.84个百分点，但新Linear也提高5.68个百分点；Gaussian相对Linear的稳定优势尚未确认。理想软件RBF为59.51%，不能称为器件成绩；拳击→拍手错误反而升至58.81%，负面结果已保留。

```bash
cd /home/ziyan/FYP_XZH
/home/ziyan/FYP_XZH/.venv/bin/python -m unittest test_kth_spatiotemporal_encoding -v
/home/ziyan/FYP_XZH/.venv/bin/python -u extract_kth_spatiotemporal.py
/home/ziyan/FYP_2026/.venv/bin/python -m unittest test_kth_spatiotemporal_selection -v
/home/ziyan/FYP_2026/.venv/bin/python -u run_kth_spatiotemporal.py
/home/ziyan/FYP_2026/.venv/bin/python verify_kth_spatiotemporal.py
/home/ziyan/FYP_2026/.venv/bin/python render_kth_spatiotemporal.py
```

提取使用现有NumPy1/OpenCV环境，训练使用现有Torch/NumPy2环境；无需安装新包。同条件完成后提取与训练直接复用，`--verify-only`只做重载检查。提取18.96秒、模型阶段14.20秒；Torch／BLAS和OpenCV算子限制1线程、未用GPU，模型进程峰值RSS约767.6 MiB。

结果：[完整说明](docs/KTH_16维时空输入对照结果.md)、[六行主结果CSV](outputs/kth_spatiotemporal_v1/primary_results_table.csv)、[清晰结果表](outputs/kth_spatiotemporal_v1/results_table.png)、[四联图](outputs/kth_spatiotemporal_v1/four_panel_summary.png)、[真实输入示例](outputs/kth_spatiotemporal_v1/figures/input_example_with_real_frame.png)。新原始16维特征、响应与Gaussian输出、模型、逐样本得分、复用记录、相同4折与Cycle清单、源码和完整耗时保存在`outputs/kth_spatiotemporal_v1/`。源Excel未找到，本轮仍为既有处理后响应库的索引留出，不是独立硬件验证。

下一步保留新输入＋G1为验证选定的软件候选，并保留G0与Linear对照；固定输入后再有限检查Gaussian校准适配及拳击退化，暂不直接进入完整硬件实验。下方各轮结果保留为阶段历史。

## 上一轮实验 Cycle感知稳健训练

研究方向：**器件响应约束下的低维视觉动作特征联合Gaussian编码与稳健识别**。已完成固定输入、固定联合Gaussian数学定义的G0／G1／G2对照。响应库Cycle索引固定240训练／80验证／80测试，视频仍为760／767／862个动作片段；没有下载或重新预处理视频，不操作硬件。

| 方案 | 测试均值响应准确率 | 测试单次Cycle准确率 | 单次Cycle Macro-F1均值 |
|---|---:|---:|---:|
| G0 均值训练 | 52.55% | 49.34% ± 0.57% | 0.4803 |
| G1 仅分类头适配Cycle，验证选定 | 52.09% | **50.75% ± 1.04%** | **0.4961** |
| G2 标准化、校准及分类头适配Cycle | 48.84% | 49.16% ± 0.43% | 0.4782 |

±为同一测试片段10轮库内抽样标准差，不是跨人置信区间。G1比G0平均提高1.42个百分点，但探索性配对区间包含0，训练8人内部4折平均排名则是G2最好；尚不能认定稳定优势。G1与G0的Gaussian特征相同，仅分类头训练响应不同。G2降低Cycle重建误差但未提高官方测试分数，负面结果同样保留。

正则化按平均损失保持一致：校准／头部原实现求和形式alpha分别为G0的0.01／0.1、G1的0.01／0.8、G2的0.08／0.8。8倍行数仍来自760段训练片段。固定目标sigma=0.25、256中心和裁剪[1e-6,1]，没有再次网格搜索。Linear与理想RBF参照条件一致，复用旧模型；G0只用训练240个Cycle重新计算均值，因此不复用旧400-Cycle模型。

原Excel未在工程内找到：当前只验证既有归一化响应库的索引留出，不能证明跨器件、跨批次或源归一化前原始测量的独立泛化。当前保留G1作软件候选；下一轮只比较仍为16维的空间与时间视觉编码，暂不直接进入完整硬件实验。

```bash
cd /home/ziyan/FYP_XZH
/home/ziyan/FYP_2026/.venv/bin/python -m unittest test_kth_cycle_robustness -v
/home/ziyan/FYP_2026/.venv/bin/python -u run_kth_cycle_robustness.py
/home/ziyan/FYP_2026/.venv/bin/python verify_kth_cycle_robustness.py
/home/ziyan/FYP_2026/.venv/bin/python render_kth_cycle_robustness.py
```

同条件已完成时直接复用，`--verify-only`只重载核验。依赖沿用[requirements_joint.txt](requirements_joint.txt)。主流程15.90秒，绘图分析约1.79秒，CPU单线程，未使用GPU；主进程实测峰值RSS约755.1 MiB。7项新增单元测试、实际模型数学检查、全部逐样本预测重载及每折训练统计审计通过。

结果：[完整说明](docs/KTH_Cycle感知稳健训练结果.md)、[表格CSV](outputs/kth_cycle_robustness_v1/results_table.csv)、[高清表格](outputs/kth_cycle_robustness_v1/results_table.png)、[组会四联图](outputs/kth_cycle_robustness_v1/four_panel_summary.png)。模型、特征、响应、Cycle与参与者划分、逐样本得分、参数、源码快照和完整耗时保存在`outputs/kth_cycle_robustness_v1/`。旧实验保留不变。

## 当前标准 原器件联合RBF流程

当前以 `/home/ziyan/SURF2026/rbf_hardware` 的原始软件流程为准：16维输入→实测16×16响应→逐维逐通道训练标准化→Ridge校准→跨维乘积重建→256维联合Gaussian→one-hot Ridge分类。原项目只读导入，未修改源码、响应库或保存模型；不复用PenDigits分类权重。**不是旧16维求和模型，输入维度有身份，不要求输入乱序后不变。** 下方HMDB51和早期KTH求和实验保留为历史分支。

KTH沿用760训练／767验证／862测试，既有16维帧差缓存不变。严格保持因子裁剪 `[1e-6,1]`，新结果为：

| 方案 | 验证准确率 | 测试准确率 | 测试Macro-F1 |
|---|---:|---:|---:|
| Linear16并训练标准化，Ridge头 | 49.80% | 44.90% | 0.4377 |
| 默认联合G | 49.41% | 47.68% | 0.4728 |
| 默认G，仅选择分类头正则化 | 53.85% | 50.12% | 0.4913 |
| 严格调整联合G | 57.50% | **52.67%** | **0.5216** |

严格配置为256个训练KMeans中心、目标 `sigma=0.25`、`calibration_alpha=0.01`、`head_alpha=0.1`、seed=42。sigma表示软件联合Gaussian目标宽度，不是改变实测器件曲线。另有裁剪上界1.2的55.45%分支，只作扩展软件诊断。参数选择仅使用官方验证部分，所有拟合只用训练部分；测试之前已查看，本轮属于探索性筛选。

已完成657个候选拟合，严格标准从现有270个合规候选选择，新增训练0次。5项数值测试、缓存对齐、599个原视频哈希、原项目来源哈希、模型重载预测与数学检查通过。原视频不重新下载，不使用GPU，不操作硬件。分类器为直接求解的Ridge，不将类别得分称为概率。

推荐研究方向：**器件响应约束下的低维视觉动作特征联合Gaussian编码与稳健识别**。当前值得继续软件研究，暂不直接进入完整硬件实验；严格G在Cycle抽样下平均49.54%，仍需稳健校准与输入表示复核。详见[完整训练与研究方向分析](docs/KTH_原器件流程训练与研究方向.md)、[组会简表](outputs/kth_joint_rbf256_strict_v1/group_meeting_table.png)、[四联图](outputs/kth_joint_rbf256_strict_v1/group_meeting_four_panel.png)、[实验说明](outputs/kth_joint_rbf256_strict_v1/实验说明.md)。

```bash
cd /home/ziyan/FYP_XZH
/home/ziyan/FYP_2026/.venv/bin/python -m unittest test_kth_joint_rbf -v
/home/ziyan/FYP_2026/.venv/bin/python -u run_kth_joint_rbf.py
/home/ziyan/FYP_2026/.venv/bin/python -u finalize_kth_joint_standard.py
/home/ziyan/FYP_2026/.venv/bin/python verify_kth_joint.py
/home/ziyan/FYP_2026/.venv/bin/python render_kth_joint_report.py
```

依赖：[requirements_joint.txt](requirements_joint.txt)。训练使用本机已有Torch环境，示例帧读取使用本项目已有NumPy1/OpenCV环境以避开二进制ABI冲突；不需要重新安装包。前两个实验命令在签名一致且已完成时直接复用；改动数据、代码、依赖须使用新目录或另建独立实验。结果与模型保存在`outputs/kth_joint_rbf256_strict_v1/`，完整候选及扩展诊断在`outputs/kth_joint_rbf256_v1/`，原始数据、模型和缓存按现有`.gitignore`留在本地。

## 以下为旧版求和分支历史记录

下文模型与结论只针对历史16维求和路线，不代替上面的当前标准。

## KTH 受控场景实验

组会文档：[视觉动作识别数据研究方向与实验结果](docs/视频动作识别_数据研究方向与实验结果.md)，以KTH为主体，汇总数据选择原因、研究方向、实验流程、对应结果及下一步建议。

### 2026年10月26日扩展调参

最新一轮复用原始16维特征，搜索8种中心布局×14种宽度×7档C，并同样扩大A的C搜索，共791个配置。21个相同旧拟合复用，新增770次拟合，全部收敛；没有重新下载或解码视频，目标仍为16核Gaussian求和、16维、峰值1、顺序无关。

验证选出的G为 `centers=linspace(0, training_P95, 16)`，实际终点 `0.47820971200638007`，`sigma=0.1`、分类器 `C=0.1`。验证准确率45.37%、Macro-F1 0.4346；测试准确率 **42.92%**、Macro-F1 **0.4176**，相对首轮调整G提高2.55个百分点／0.0296，但仍低于A的47.91%／0.4680。默认Gaussian只扩大C搜索则选到C=0.001，验证提高但测试降为37.47%；完整六行对比保留该结果，不根据测试改选。

输入和全部预处理参数仍只拟合训练8人，先锁定参数再测试；现有验证／测试此前均已查看，属于探索性软件筛选，不是独立最终验证。本轮CPU墙钟51.42秒，新增拟合合计37.44秒，未使用GPU。当前仍不支持直接进入硬件实验。

```bash
.venv/bin/python -m unittest -v test_kth.py test_tune_kth_gaussian.py
.venv/bin/python -u tune_kth_gaussian.py
.venv/bin/python tune_kth_gaussian.py --verify-only
.venv/bin/python tune_kth_gaussian.py --render-only
```

结果：[扩展调参说明](outputs/kth_gaussian_expanded_v2/扩展调参说明.md)、[完整对比表](outputs/kth_gaussian_expanded_v2/results_table.png)、[验证参数热图](outputs/kth_gaussian_expanded_v2/validation_parameter_heatmap.png)。逐配置指标、模型、预测、特征映射和断点缓存均保存在 `outputs/kth_gaussian_expanded_v2/`；相同已完成条件直接复用，变更搜索条件请指定新的 `--output`。

### KTH首轮实验与复现

新增 KTH 六类 `walking/jogging/running/boxing/handwaving/handclapping`，作为简单背景下的输入表示对照；原 HMDB51 代码与结果保留不变。仍然使用中间1秒的17帧、64×64灰度全图平均绝对帧差，得到16维输入；Gaussian每核对16个输入响应求和，最终16维，不引入CNN、光流、骨架或硬件操作。

数据来自 [KTH官方发布页](https://www.csc.kth.se/cvap/actions/)。使用其 `00sequences.txt` 动作片段边界与原始参与者划分：8人训练、8人验证、9人测试，同人/同视频片段不会跨集合。官方标注含599个视频、2391个动作片段，另有一个明确标记缺失的视频；实际纳入数量以运行清单为准。

```bash
# 在本项目根目录运行；复用既有 .venv 和 requirements.txt
.venv/bin/python download_kth.py
.venv/bin/python -m unittest -v test_kth.py
.venv/bin/python -u run_kth.py
.venv/bin/python verify_kth.py
```

Gaussian有限候选为中心均匀 `[0,1]` 或训练输入P1–P99范围，统一宽度 `0.05/0.1/0.2`；A、G默认、G调整均比较分类器 `C=0.1/1/10`。只用官方验证集Macro-F1选择配置，准确率辅助；固定配置后评价测试。输入缩放、数据相关中心和StandardScaler都只拟合训练8人的数据，最终不合并验证集重训。这里采用官方单次验证划分，不声称3折均值或误差线。

新代码：`download_kth.py`、`run_kth.py`、`test_kth.py`、`verify_kth.py`。数据另存 `data/kth/`，结果另存 `outputs/kth_six_subject_split_v1/`。缓存复用检查视频内容、清单、代码、划分与参数；已完成且条件一致时不重复训练。条件变更应使用新的 `--output`，仅重绘可运行：

```bash
.venv/bin/python run_kth.py --render-only
```

结果入口：[结果表](outputs/kth_six_subject_split_v1/results_table.csv)、[四联图](outputs/kth_six_subject_split_v1/four_panel_summary.png)、[中文实验说明](outputs/kth_six_subject_split_v1/实验说明.md)。数据下载归档、原视频、特征缓存和模型保留本地，不随Git上传。KTH数据仅限官方允许的非商业用途，发表时请引用 Schuldt, Laptev and Caputo, *Recognizing Human Actions: A Local SVM Approach*, ICPR 2004。

本机已运行完成，实际纳入 **760训练／767验证／862测试**，均为动作子片段而非独立视频文件。两段排除原因分别为采样解码失败、短于1秒。

| 方案 | 训练准确率 | 验证准确率 | 测试准确率 | 测试 Macro-F1 |
|---|---:|---:|---:|---:|
| A 原始16维输入 | 53.82% | 56.06% | 47.91% | 0.4680 |
| G 默认 | 40.92% | 39.77% | 40.26% | 0.3864 |
| G 调整 | 36.58% | 43.29% | 40.37% | 0.3880 |

验证选定的G为16个中心均匀覆盖训练输入P1–P99，即 `[0.0317144463, 0.6815130554]`，`sigma=0.2`、分类器 `C=0.1`；A和G默认的分类器也选到 `C=0.1`。调整G相对默认G只多识别对1个测试片段，仍低于A 7.54个百分点，暂不支持直接进入硬件实验。下一步优先验证输入表示，而非继续只扫描Gaussian宽度；完整解释见[结果解读与建议](outputs/kth_six_subject_split_v1/结果解读与建议.md)。

实测：官方下载、校验和解压墙钟约10.92分钟；全量采样预处理14.86秒，21次分类器拟合合计0.464秒，实验墙钟25.65秒（不含下载）。CPU运行，未用GPU，实验进程峰值RSS 614.1 MiB，包含图表生成；不将采样帧概念存储量当作峰值内存。保存的字符串NPZ数组及joblib模型仅应作为本机可信实验产物加载；若访问NPZ的字符串元数据，需要 `np.load(..., allow_pickle=True)`。

## HMDB51 实验（历史结果保留）

本项目使用真实 HMDB51 视频，研究一个简单问题：**将视频转换为 16 个画面变化量后，固定 Gaussian 映射和核内求和是否能改善动作分类，是否值得继续开展硬件实验。**

当前仅完成软件实验，未操作硬件、未模拟未知器件动态，也未使用 CNN、光流或骨架模型。实验类别为 `walk`（走路）、`run`（跑步）、`jump`（跳跃）、`sit`（坐下）、`stand`（起立）、`wave`（挥手）。

## HMDB51 最近一轮结果

采用 HMDB51 官方 split 1。六类原有 420 个训练、180 个测试候选；成功读取并满足采样条件的样本为 **412 个训练、176 个测试**，排除原因保存在[排除记录](outputs/hmdb51_software_feasibility_v3/exclusions.csv)。

最新一轮使用相同的固定分层 3 折，仅在官方训练部分选参数，种子为 42，以验证 Macro-F1 均值为主要依据。下表中的三组分类器均独立比较正则化 `C=0.1/1/10`。

| 方案 | 输出维度 | 训练准确率 | 验证准确率（均值 ± SD） | 验证 Macro-F1 | 测试准确率 | 测试 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| A：原始输入 | 16 | 27.43% | 17.47% ± 1.88% | 0.1626 | 17.05% | 0.1538 |
| G 默认 | 16 | 25.97% | 18.94% ± 5.15% | 0.1598 | 18.75% | 0.1604 |
| G 调整 | 16 | 25.49% | 19.66% ± 1.98% | 0.1814 | 18.18% | 0.1752 |

- A 的分类器 `C=0.1`。
- G 默认：16 个中心均匀覆盖 `[0,1]`，`sigma=0.1`，分类器 `C=10`。
- G 调整：16 个中心均匀覆盖训练输入的 P1～P99；完整训练拟合后范围约为 `[0.000290541, 0.250112758]`，`sigma=0.05`，分类器 `C=10`。
- 三组均为 `StandardScaler + LogisticRegression`，`solver=lbfgs`、`max_iter=5000`、`tol=1e-6`。G 默认的 Gaussian 参数未变，但分类器 C 经过同样规则选择，因此其成绩不等于最早的 B（C=1）。

**当前16维输入与所测高斯方案效果不足，暂不支持直接进入硬件实验。** G 调整相对 A 只有小幅改善，相对 G 默认未提高准确率；所选 G 的 30 个 walk 测试样本全部错分。软件候选配置已保存，后续优先检查输入是否混合了动作、背景活动与镜头变化。

![最新软件可行性四联图](outputs/hmdb51_software_feasibility_v3/four_panel_summary.png)

结果入口：

- [结果表 CSV](outputs/hmdb51_software_feasibility_v3/results_table.csv) / [高清 PNG](outputs/hmdb51_software_feasibility_v3/results_table.png)
- [简短中文说明](outputs/hmdb51_software_feasibility_v3/简短中文说明.md)
- [完整软件筛选说明](outputs/hmdb51_software_feasibility_v3/软件筛选说明.md)
- [模型核对说明](outputs/hmdb51_software_feasibility_v3/模型核对说明.md)
- [排列不变性检查](outputs/hmdb51_software_feasibility_v3/order_independence_checks.json)
- [逐样本预测与概率](outputs/hmdb51_software_feasibility_v3/sample_predictions.csv)
- [各类指标](outputs/hmdb51_software_feasibility_v3/per_class_metrics.csv)

## 输入表示与数学模型

每段视频取中间固定 1 秒窗口，按实际 FPS 等间隔定位 17 帧。每帧先转灰度、缩放至 `64×64`，再转为 `float32` 并除以 255。相邻两帧的逐像素绝对差取均值，得到 16 维输入：

```text
x_i = mean(abs(frame_(i+1) - frame_i)), i = 1,...,16
```

用相应训练部分所有输入数值的全局最大值统一缩放。验证/测试沿用该缩放参数，并截断超出 `[0,1]` 的值；不按视频或输入位置分别归一化。帧时间会量化到离散帧，实测采样跨度与重复索引见[采样检查](outputs/hmdb51_diagnostics_v2/sampling_span_and_repeated_indices.csv)。

第 j 个核对每个标量输入产生峰值为 1 的 Gaussian 响应，再将该核的全部 16 个响应相加：

```text
R[n,i,j] = exp(-(X[n,i] - mu[j])**2 / (2*sigma**2))
Y[n,j]   = sum_i R[n,i,j]

X: [N,16] -> R: [N,16,16] -> Y: [N,16]
```

每核只输出一个合计值，最终仍为 **16 维**。交换输入位置不会改变这个和；重复输入按出现次数累加，不去重。当前软件中的中心和宽度使用归一化输入坐标，没有擅自赋予脉冲时间或电流单位。

最早的 B 与这个目标定义一致。历史 C 将中间响应按输入位置展开为 256 维，是位置保留的软件诊断；它不满足本轮顺序无关要求，其成绩不用于判断目标模型的硬件可行性。

## 工程结构

| 文件或目录 | 用途 |
|---|---|
| `run_experiment.py` | 视频采样、16维特征缓存、默认 A/B 对比 |
| `download_hmdb51.py` | 官方视频和 split 下载、六类解压辅助脚本 |
| `tune_gaussian.py` | 历史5折参数探索：宽度与中心策略 |
| `diagnose_hmdb51.py` | 实现审计、训练表现、3折选参、历史 A/B/C 诊断 |
| `verify_hmdb51_diagnostics.py` | 独立核对保存的特征、模型预测和指标，不重新拟合 |
| `software_feasibility.py` | 顺序无关目标模型核对、有限选参、缓存复用和汇报图 |
| `requirements.txt` | 经实验使用的固定版本依赖 |
| `outputs/` | 已完成实验的小体积指标、图表、配置、预测及源代码快照 |
| `docs/README_历史实验记录.md` | 本次整理前的完整 README，保留历史说明 |

## 安装

实测环境为 Python 3.10.12、NumPy 1.26.4、OpenCV 4.11.0、scikit-learn 1.7.2。全部实验使用 CPU，无需 GPU。

```bash
git clone https://github.com/Zhi246/FYP.git
cd FYP
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

不要直接依赖系统已有的 OpenCV/NumPy；本项目固定 `numpy==1.26.4` 和兼容的 OpenCV 版本，以避免预装二进制版本冲突。

## 准备数据

原始视频、下载归档和虚拟环境不随 Git 仓库上传。数据来自 [HMDB51 官方页面](https://serre.lab.brown.edu/hmdb51.html)，请按其使用要求获取数据。程序递归寻找类别视频及 split 文件，典型目录为：

```text
data/
├── videos/hmdb51/
│   ├── walk/*.avi
│   ├── run/*.avi
│   ├── jump/*.avi
│   ├── sit/*.avi
│   ├── stand/*.avi
│   └── wave/*.avi
└── splits/
    └── testTrainMulti_7030_splits/
        ├── walk_test_split1.txt
        ├── run_test_split1.txt
        └── ...
```

官方标记 `1` 用于训练，`2` 用于测试，`0` 排除。少于1秒或无法完成解码/采样的视频跳过并记录原因。

若使用官方 RAR 下载辅助脚本，系统需要 `unrar`。也可在 Ubuntu 中将解压工具安装到项目目录而不修改系统：

```bash
mkdir -p data/tools
cd data/tools
apt download unrar
dpkg-deb -x unrar_*.deb .
cd ../..
.venv/bin/python download_hmdb51.py --unrar data/tools/usr/bin/unrar-nonfree
```

历史运行中官方 Google Drive 下载受到访问限制，实际使用官方页面列出的 Hugging Face 镜像获取视频 ZIP 和 split ZIP；来源与实际耗时保存在[下载记录](outputs/hmdb51_six_split1/download_extract_timing.json)。解压后的目录符合上面结构即可；不要用随机数据代替缺失的视频。

## 复现实验

只阅读已提交的结果和图表无需下载数据。首次完整复现需要准备真实数据，并依次生成本地特征缓存、模型和前序验证结果：

```bash
# 1. 原始 A/B：每个视频只提取一次特征
.venv/bin/python run_experiment.py \
  --video-root data/videos --split-root data/splits

# 2. 历史参数探索，生成诊断审计所需的前序结果
.venv/bin/python tune_gaussian.py

# 3. 实现检查与统一3折诊断
.venv/bin/python diagnose_hmdb51.py --stage audit
.venv/bin/python diagnose_hmdb51.py --stage experiment
.venv/bin/python verify_hmdb51_diagnostics.py

# 4. 最新有限软件筛选：另存本地目录，保留已提交的历史报告
.venv/bin/python software_feasibility.py \
  --output outputs/hmdb51_software_feasibility_local
```

这些命令会重建本地缓存与模型，并更新相应前序实验目录。已提交表格是历史实测记录，新运行的路径和耗时可能不同。`run_experiment.py` 支持 `--rebuild-cache`，日常运行优先复用缓存。

如果已经保留本机原实验的全部数据和缓存，可直接运行 `software_feasibility.py`；只有数据、模型、划分、缩放、参数及实现一致时才复用结果。最新一轮实际复用63个折内拟合，未重跑交叉验证，只新增2个最终模型拟合。重复运行会检查签名；条件不一致时应指定新的输出目录。

仅重绘已完成的本地结果：

```bash
.venv/bin/python software_feasibility.py \
  --output outputs/hmdb51_software_feasibility_local --render-only
```

Git 忽略了 `data/`、`.venv/`、Python 缓存、数值特征缓存（`.npz/.npy`）、模型（`.joblib/.pkl`）、重复备份及大体积采样接触图。这些文件保留在本机，可通过以上命令重建。已提交的 `source_code/` 是历史代码快照，用于实现身份核对，请勿将它当作最新执行入口。

## 实验边界

- 测试集已在多轮探索中查看，本项目属于探索性软件筛选，不称为独立最终验证。
- 没有经过验证的同源视频 ID 表；最新比较沿用相同的分层3折，可能有同源/同场景片段跨折。历史5折文件名分组与最新协议分开记录。
- 每折的输入缩放、数据相关中心和 StandardScaler 都只在该折训练部分拟合，正式测试不参与选参。
- 每个目标 G 对全部588个样本各做100次排列检查，特征、概率均在记录容差内一致，预测改变为0；重复值累加检查通过。
- A/G 的差别同时包含 Gaussian 映射与求和，不能全部归因于非线性。历史256维C的差别还涉及位置保留、维度、表达能力和正则化。
- 当前结果仅适用于所选六类、当前输入与所测配置，不代表器件不可行，也不代表所有视频任务或整个 HMDB51 永久不可用。

## 数据集引用

H. Kuehne, H. Jhuang, E. Garrote, T. Serre, and T. Poggio. *HMDB: A large video database for human motion recognition*. ICCV, 2011. 视频数据及其使用要求以原发布方为准。
