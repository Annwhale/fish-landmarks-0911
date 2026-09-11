![Welcome, and thank you for reviewing my data and code. Have a wonderful day!](assets/welcome-0911.gif)

# Fish landmarks · 0911

鱼类外部形态点数据的清理、坐标转换、验证与绘图代码。对应资源包含2,564张去重图像、11个点位角色和2,555张技术验证图像。本仓库不是数据存储库，也不代表论文已被接收。

## 安装与检查

Python 3.11。在已有 PyTorch 环境中运行：

```bash
python -m pip install -e .
python -m pip install pytest scipy scikit-learn seaborn
python -m pytest -q
python run-0911.py --help
```

`environment.yml`记录实验依赖配置；本次验证使用已有实验环境，不声称已经在所有平台完成干净安装。NVIDIA LocateAnything 可选依赖见 `pyproject.toml` 中的 `locateanything`。GPU训练、下载权重不是单元测试的一部分。

无数据时预期19项测试通过、13项数据依赖测试明确跳过，不是32项全通过。接入完整作者包后再运行测试；四项源格式测试还需把 `FISH_SOURCE_ROOT` 环境变量设为原始采集目录。

## 数据接入

原始图像和第三方权重不在 Git 中。主数据的公开访问地址尚未发布；在此之前，第三方无法仅凭本仓库完整复现数据依赖的分析。若已有完整作者包，解压后运行：

```bash
python run-0911.py attach /path/to/FishLandmarks_Scientific_Data_Submission_v1
python run-0911.py quickstart data/benchmark_v1
```

`attach`只建立指向已有数据和验证资料的软链接，遇到不一致路径会停止，不覆盖已有文件。也可将独立发布的 `benchmark_v1` 放到 `data/benchmark_v1`。

从原始采集目录重建需先修改 `config/dataset.yaml` 的源目录及数据库路径；旧审计包含重复图检测，不必为常规阅读或复用重复运行。论文处理代码保留这些功能，以保证可复现性。

## 使用

公开入口采用连字符，内部模块保留导入和测试所需命名。

```bash
python run-0911.py train-yolo --help
python run-0911.py train-heatmap --help
python run-0911.py evaluate-yolo --help
python run-0911.py evaluate-heatmap --help
python run-0911.py build-tex manuscript.md manuscript.tex --standalone
```

完整入口映射见 `run-0911.py`。绘图脚本位于 `figures/scripts`，数值源数据位于 `figures/source_data`。部分绘图和外部验证脚本还需完整验证资料及按上游许可单独获取的 Melops 数据；运行前检查对应脚本的路径和输入。附带数值表不构成生物学金标准或 SOTA 保证。

## 点位解释

见 `docs/landmark-definitions-0911.csv`。历史字段 `canonical_name_provisional` 是冻结别名；第10点不能直接当作胸鳍附着点，第6点不是计算中点。此次定义补充没有移动任何标注坐标，也没有重算已冻结模型性能。原始标注协议的精确操作规则仍需团队最终签认。

## 许可与工具说明

项目代码按 MIT 发布。依赖、外部数据和权重遵循各自许可，见 `docs/third-party-notices.md`。本仓库不包含第三方权重、原图、私人作者资料或历史稿件。

OpenAI Codex 辅助了代码开发、数据审计和文稿整理；此说明不将工具输出视为团队测量数据，也不替代作者的科学审核。代码注释采用简短中文，`><(((o>` 为项目标记。
