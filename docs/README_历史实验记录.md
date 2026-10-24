# HMDB51 六类 Gaussian 求和预实验

实验类别固定为 `walk, run, jump, sit, stand, wave`，使用官方 split 1。A 是缩放后的 16 维相邻帧差；B 对每个标量施加 16 个固定 Gaussian 后沿输入位置求和，输出仍为 16 维。两组分类器设置完全相同。

## 复现

```bash
cd /home/ziyan/FYP_XZH
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 如系统没有 unrar，可无 root 安装到项目中：
mkdir -p data/tools
cd data/tools && apt download unrar && dpkg-deb -x unrar_*.deb . && cd ../..

.venv/bin/python download_hmdb51.py --unrar data/tools/usr/bin/unrar-nonfree
.venv/bin/python run_experiment.py
```

再次运行会复用 `outputs/hmdb51_six_split1/feature_cache.joblib`。强制重建使用 `--rebuild-cache`。如果数据已在别处，可传 `--video-root` 和 `--split-root`。

## 输出

`outputs/hmdb51_six_split1/` 包含：指标 CSV、A/B 混淆矩阵 PNG/CSV、逐样本预测、原始/缩放/Gaussian 特征、缩放参数、模型、完整清单、排除原因、输入示例及训练集分布图、运行环境与数据成本统计，以及中文实验说明。

采样通过实际 FPS 将视频中间 1 秒映射到 17 个等间隔时间点，不会直接取 17 个相邻原始帧。所有缩放与 StandardScaler 均只在训练数据上拟合。

## Gaussian 宽度与中心参数实验

```bash
cd /home/ziyan/FYP_XZH
.venv/bin/python tune_gaussian.py
```

复用原始特征缓存，在官方训练部分内按文件名推断同源视频，进行5折分组验证。比较 8 种宽度 × 6 种中心分布（共48组），每折的全局最大值、数据相关中心和 StandardScaler 仅在该折训练数据上拟合。验证平均准确率决定配置，冻结选择后才评估官方测试集。

新结果保存到 `outputs/hmdb51_gaussian_tuning_v1/`，包含全部验证结果、宽度曲线、中心×宽度热图、最终模型/中心/特征/逐样本预测及中文说明。原来默认 A/B 目录保留。官方测试集在上一轮已经查看过，这是后续探索性实验。

## 实现检查与 A/B/C 低准确率诊断

```bash
cd /home/ziyan/FYP_XZH
.venv/bin/python diagnose_hmdb51.py --stage audit
.venv/bin/python diagnose_hmdb51.py --stage experiment
.venv/bin/python verify_hmdb51_diagnostics.py
```

新目录为 `outputs/hmdb51_diagnostics_v2/`。先备份两轮旧结果及代码，核查文件/标签/官方划分和特征行对齐、训测内容重复、实际采样跨度、浮点差分、Gaussian公式与求和轴、训练拟合的预处理和收敛情况。从每类按文件名排序选2段成功读取的训练视频，独立顺序解码核对原缓存并保存17帧彩色/灰度接触图。检查失败或缓存改变时停止后续比较。

本轮按最新实验指令使用固定种子42的分层3折：由于没有经验证的同源ID表，不使用上轮文件名启发式分组。A与B都比较分类器C=0.1/1/10。B的中心分别是均匀[0,1]、均匀训练P1至P99、训练1%至99%等间隔分位点；σ=0.01/0.02/0.05/0.1/0.2。每折从原始输入重新拟合该折训练预处理，以验证Macro-F1均值主选。

C为256维软件诊断：使用B从训练验证选出的中心策略和σ，保留全部16×16响应，按输入位置优先展开，分类器C也用同样规则选择。所有参数锁定后才进行本轮最终测试评估。完整记录见新目录的 `诊断实验说明.md`。

## 顺序无关目标模型的软件可行性筛选

```bash
cd /home/ziyan/FYP_XZH
.venv/bin/python software_feasibility.py
```

输出为 `outputs/hmdb51_software_feasibility_v3/`。目标模型是旧B：16个标量输入分别经每核Gaussian响应后在核内求和，每核输出1值，总计16维。交换输入顺序不改变输出。旧C是位置保留诊断，不满足本轮顺序无关目标，本轮不训练C、不用它的分数判断目标可行性。

只比较默认均匀[0,1]中心和训练P1至P99均匀中心，σ=0.05/0.1/0.2，分类器C=0.1/1/10。A也比较同样的分类器C；G默认固定Gaussian参数，但C仍由训练交叉验证选择，因此不必等于旧B(C=1)。以验证Macro-F1均值选参，完整保存并锁定后才评价测试集。

已核对原始缓存、官方划分、旧视频内容哈希、数学实现代码AST、缩放、中心、分层3折、分类器设置和依赖后，复用匹配的63个CV拟合，本轮CV新训练数为0。只补拟合缺失的最终模型。对每个目标G的全588样本各做100次排列，检查特征、概率和类别，并独立核查重复值的重数累加。不操作硬件、不下载、不更改预处理。

交付的 `results_table.csv/png`、`four_panel_summary.png`、四张独立子图、`简短中文说明.md`、`模型核对说明.md` 可用于组会。复次运行会比较数据/参数签名，条件一致时直接返回已完成结果；如需重绘而不训练，使用 `--render-only`。
