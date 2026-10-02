# Protenix 基线记录

本机工作目录：`/home/husrcf/Code/fireprot`
DiamondHill 运行目录：`pc@10.120.16.9:/media/IntelSSD/fireprot`

## 已准备的内容

- `Protenix/`：官方仓库的完整 Git clone，非 shallow，所有获取到的分支／tag与历史保留。
  上游 `https://github.com/bytedance/Protenix.git`。
  当前工作分支 `fireprot/baseline`，以 `v1.1.0` 为起点：
  `d3b4db6a121dd4584edd85e93744239325a2b72e`，对应现有环境的 Protenix 1.1.0。
  clone 时的上游 main 为 `85767b811c40ed46e73a9b39519cf6bfca8701ba`；可以另外开分支研究。
- `cases/`：8个已有开发单链案例，124–164 aa。各有官方格式 `input.json`、FASTA、
  `gt_mapping.npz` 与来源 `metadata.json`；`all_inputs.json` 可一次跑全部。
  包含1L12/1L04、1TAY/1TDY、1GOB/2RN2、1IZR/1IZQ四对。
- `scripts/infer.sh`：显式从本工作区源码运行，默认 Mini-ESM、C4/S1、一个样本、FP32、
  seed101、关闭外部MSA/template搜索，使用ROCm可运行的torch triangle kernels。
- `scripts/sync_to_diamondhill.sh`：同步本机源码、案例及脚本；保留远端权重与输出。
- `provenance/`：源码版本、案例SHA256、权重/CCD来源与校验清单，以及运行自检结果。

权重与较大的运行资源按约定放在 **DiamondHill**，本机保留源码、案例和清单：

|远端 `runtime/checkpoint/` 文件|大小（十进制）|用途|
|---|---:|---|
|`protenix_mini_esm_v0.5.0.pt`|0.542 GB|当前主线使用的公开 Mini-ESM|
|`protenix_mini_default_v0.5.0.pt`|0.537 GB|公开 Mini，无 ESM 依赖的对照|
|`protenix_tiny_default_v0.5.0.pt`|0.443 GB|更小模型的工程对照|
|`esm2_t36_3B_UR50D.pt`|5.678 GB|Mini-ESM 需要的 ESM2-3B|
|`esm2_t36_3B_UR50D-contact-regression.pt`|6.8 KB|ESM 配套文件|

`runtime/common/` 包含CCD mmCIF及RDKit缓存，初始资源共7.833 GB。
这些是独立文件副本（可能使用文件系统CoW），不是指向旧实验的软链接或硬链接。
复制后逐文件SHA256一致。原有微调的fixed/diverse未混入这个公开权重基线；
full/base模型权重未加入本次小型启动包。

## 开始运行

先在本机修改 `Protenix/` 内的源码，然后：

```bash
cd /home/husrcf/Code/fireprot
./scripts/sync_to_diamondhill.sh
ssh pc@10.120.16.9
cd /media/IntelSSD/fireprot
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 ./scripts/infer.sh \
  --input /media/IntelSSD/fireprot/cases/1izr_L124/input.json \
  --out_dir /media/IntelSSD/fireprot/outputs/my_first_run
```

用新的输出目录，便于保留不同实验的产物。`--step 5` 可改为C4/S5；
`--model_name protenix_mini_default_v0.5.0` 或
`--model_name protenix_tiny_default_v0.5.0` 可换基础模型。
调用8例时改为 `--input /media/IntelSSD/fireprot/cases/all_inputs.json`。

远端复用已安装环境 `/home/pc/anaconda3/envs/fold/bin/python`，不修改共享环境安装。
脚本设置 `PYTHONPATH` 优先读取这里的 `Protenix/`，并设置
`PROTENIX_ROOT_DIR=/media/IntelSSD/fireprot/runtime`、`LAYERNORM_TYPE=torch`。
因此本机源码同步后的修改会进入运行；不依赖 onestepfold 的实验脚本或 PRoot 路径映射。
如果换Python，显式设置 `FIREPROT_PYTHON`，先确认兼容依赖。

完整可用入口还在 `Protenix/runner/`，原生示例与文档在 `Protenix/examples/`、
`Protenix/docs/`。主要源码入口是 `Protenix/protenix/model/protenix.py`、
`Protenix/protenix/model/modules/pairformer.py` 和
`Protenix/protenix/model/modules/diffusion.py`；这只是源码导航，不是已完成的热点分析。

## 案例和验证的口径

GT文件是按原生原子身份映射的实验坐标，含 `coordinates`、`mask`、`atom_names`、
`residue_ids`、`chain_ids` 和 `reference`。`reference` 是模型输入参考构象，不是GT；
只有 `coordinates[mask]` 是观测实验坐标。不要将缺失位置当零坐标评分。
病例已参与方法开发；不是独立测试集，也不是长链性能基准。
1TAY/1TDY及1IZR/1IZQ含潜在二硫键接触，原生单序列输入不自动补全该拓扑。

本工作区的原生CLI使用其自身sampler、增强及dropout设置，不等同于旧研究报告中的
受控确定性sampler；旧报告坐标不能直接作为这条CLI的逐位正确性金标准。
`provenance/smoke_report.json` 检查三个公开模型能从这里加载并输出有限坐标，
仅为安装／资源自检，不代表性能提升、精度或可微性验收。

后续加速比较应先固定模型、输入、dtype、cycle、step、sample和随机状态，分别记录
首次加载、预处理/ESM、Pairformer、diffusion以及端到端耗时。当前不启动训练或加速改动。

## 版本与独立性

远端 `provenance/runtime_snapshot/` 额外保存已安装 Protenix/runner/configs 源码供追溯，
它不是默认执行代码。`Protenix/.git` 在两端均完整保留。
同步脚本不覆盖远端 `.git`，不自动commit或push；它只复制工作树，远端可用git diff核对。
本机这批新sidework文件独立于 `/home/husrcf/Code/onestepfold`。

## 本次自检结果（2026-10-02）

三个模型均在 `1izr_L124` 上跑通C4/S1、FP32、sample1、seed101，
各输出945个原子且坐标全部有限。整个并行自检墙钟约68秒，包含初始化，不能作为加速比。
`outputs/` 保留本次CIF和置信度JSON，本机也复制了这些小型结果。
官方CLI另外下载了所需common索引；远端完整运行资源清单见
`provenance/runtime_manifest.json`，初始7个已核验副本见 `assets.json`。

原生CLI会在 `Protenix/esm_embeddings/` 写入ESM缓存，该目录已列入本地Git exclude；
做冷／热启动比较时要明确缓存命中条件。缓存不是源码修改，也未作为独立性能结果。

## 顶层仓库与协作记录

工作区根目录另有独立 Git 仓库，管理根目录 README、`cases/`、`scripts/`、
`provenance/`、`AGENTS.md` 和 `.agents/`。`Protenix/` 保留自己的完整 Git 历史，
不作为顶层仓库的子模块；在该目录内分别查看或提交源码改动。
顶层远端 `origin` 为 `https://github.com/Media-Foundry/FireProt.git`。
顶层 `.gitignore` 排除 `Protenix/`、`runtime/` 和 `outputs/`。
项目背景和实验口径见 `.agents/project_context.md`，仓库划分见
`.agents/decision_log.md`。
