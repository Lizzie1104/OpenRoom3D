# OpenRoom3D：3D 视觉课程项目

用一段桌面视频重建 3D 高斯场景，通过文字查询选出物体，再用拖动控件或自然语言完成刚体编辑。

## 新功能：自然语言 3D Agent

在查看器中输入：

> 把笔记本向右移动一步，沿桌面旋转 90 度，检查碰撞，最后保存场景。

Agent 使用 **DeepSeek Flash（`deepseek-flash`）**，每次请求显式关闭思考模式：
`thinking: {"type": "disabled"}`。模型生成结构化操作计划，确定性的几何模块执行变换，不让模型直接修改高斯参数或运行代码。

- **物体选择与编辑：** 选择已加载候选、增量移动、绕世界轴或桌面法向旋转、复位和聚焦。
- **多物体操作：** 切换选择保留各物体的编辑；可将一个物体摆放到另一个物体附近。
- **碰撞检查：** 对已加载的独立物体进行包围盒检查；变换前先模拟整条计划，目标位置碰撞时拒绝执行。
- **场景保存：** 导出包含各物体变换的完整高斯场景 NPZ 和位姿说明 JSON，保持原始输入文件。
- **显示控制：** 用语言切换隔离、热力图、坐标轴、包围盒和已有场景开关。

本次扩展沿用原有 CLIP 查询、SAM 2 分割、重建、训练和刚体变换算法。完整工具说明见 [语言 Agent 使用指南](OpenRoom3D/docs/language_agent.md)。

## 快速体验

先克隆仓库，进入代码目录：

```bash
git clone https://github.com/Lizzie1104/OpenRoom3D.git
cd OpenRoom3D/OpenRoom3D
```

查看器需要 Python、NumPy、SciPy、Matplotlib 和支持高斯预览的 Viser。已有环境可直接使用；
新环境的配置及训练依赖见 [环境说明](OpenRoom3D/docs/setup.md)。保存好的演示场景与物体 NPZ 已在仓库中，打开这个演示不需要重新训练或下载 checkpoint。

### Windows / PowerShell

在同一个终端设置你自己的 DeepSeek 密钥并启动：

```powershell
$env:DEEPSEEK_API_KEY = '你的 DeepSeek API Key'
python scripts/object_pose_viewer.py --scene outputs/retrain_2026-11-01/selection_scalereg/scene.npz --object 'Laptop=outputs/retrain_2026-11-01/selection_scalereg/obj_laptop.npz' --plane outputs/retrain_2026-11-01/inferred_table/plane.npz --inferred-background outputs/retrain_2026-11-01/inferred_table/inferred_background.npz --port 7013
```

### Linux / Bash

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek API Key'
bash scripts/start_retrain_demo.sh
```

启动脚本默认使用已有 viewer 环境；也可执行 `PYTHON=/path/to/python bash scripts/start_retrain_demo.sh` 指定 Python。

打开 [http://localhost:7013](http://localhost:7013)，展开 **自然语言 3D Agent**，输入指令并点击 **执行指令**。
密钥只从 Python 进程环境读取。不配置密钥也能使用原有手动控件。按 Ctrl+C 结束查看器。

### 使用约定

- 默认演示只加载笔记本。操作水杯等其他物体前，需通过现有查询流程准备同一场景的候选，并用多个 `--object LABEL=OBJECT_NPZ` 参数加载；候选不明确时 Agent 会询问用户。
- 右、前、上分别对应世界坐标 +X、+Y、+Z，与相机视角无关。距离使用**重建单位**，没有厘米或米的标定；未给距离时使用界面的 Translation step。
- 贴桌约束默认开启；抬高或倾斜需关闭它。切换候选保留编辑，Reset object 只复位当前物体。同一物体的重叠分割候选不能同时编辑。
- 碰撞检测是高斯中心包围盒的粗略检测，只覆盖已加载的独立物体和操作后的目标位置，不覆盖移动轨迹、未分割背景或桌面边缘。
- 场景默认保存到 `outputs/object_edits/scene_<时间>_edited.npz` 及同名 JSON，不包含推断桌面层、显示开关或训练 checkpoint。原有物体导出按钮仍只导出选中物体。

## 验证

在 `OpenRoom3D/` 代码目录、配置好查看器依赖后运行：

```bash
python -m pytest tests/test_language_agent.py tests/test_pose_viewer.py tests/test_object_geometry.py -q
```

相关测试覆盖多物体变换、桌面约束、碰撞拒绝、整场景导出、原始数据保留和 Flash 非思考请求格式。
自动化测试使用固定的网络响应；实际语言操作需要有效的 DeepSeek 密钥及网络连接。

## 项目文档

- **当前成果与下一步路线：** [OpenRoom3D/docs/status_and_roadmap.md](OpenRoom3D/docs/status_and_roadmap.md)，接手前请先读这份。
- **项目说明与目录结构：** [OpenRoom3D/README.md](OpenRoom3D/README.md)
- **全部实验数据：** [OpenRoom3D/docs/experiments.md](OpenRoom3D/docs/experiments.md)

## 仓库里有什么

| 位置 | 内容 |
|---|---|
| `OpenRoom3D/` | 代码、测试、文档、相机数据、两组 101 帧照片、SAM 2 掩码、实验结果和对比图 |
| `training-assets/models/sam2/repo/` | SAM 2 源码（Apache-2.0） |
| GitHub Release `assets-2026-11-02` | 8 个模型 checkpoint 和 1 GB 的语义特征缓存 |
| 官方下载地址 | CLIP、SAM 2.1、LPIPS 预训练权重；地址和校验值见 `OpenRoom3D/docs/large_files.json` |

## 下载大文件

超过 GitHub 单文件大小限制的文件不在 git 里。克隆后在 `OpenRoom3D/` 目录下运行下面的命令，脚本会把文件放到原来的位置，并逐个校验 SHA-256：

```bash
python scripts/fetch_assets.py --convert-clip
```
