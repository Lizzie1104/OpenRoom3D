# 自然语言 3D Agent

在原有 Viser 查看器中展开 **自然语言 3D Agent**，输入中文或英文指令并点击 **执行指令**。
仅新增语言操控及其必需的多物体编辑状态；重建、CLIP 查询、SAM 2 分割、训练和刚体变换算法保持原有实现。

## 启动

模型固定为 `deepseek-flash`，请求固定携带 `thinking: {"type": "disabled"}`，没有思考模式切换。
密钥从 Python 进程环境读取，不进入网页、场景文件或导出记录。不需要安装额外的语言模型 SDK。

PowerShell（在 OpenRoom3D 目录中）：

```powershell
$env:DEEPSEEK_API_KEY = '你的 DeepSeek API Key'
python scripts/object_pose_viewer.py --scene outputs/retrain_2026-11-01/selection_scalereg/scene.npz --object 'Laptop=outputs/retrain_2026-11-01/selection_scalereg/obj_laptop.npz' --plane outputs/retrain_2026-11-01/inferred_table/plane.npz --port 7013
```

使用原来的 Linux 演示入口时，先在同一终端执行 `export DEEPSEEK_API_KEY='你的密钥'`，再执行
`bash scripts/start_retrain_demo.sh`。打开 http://localhost:7013 。不设置密钥也可以使用原来的手动控件。
查看器依赖和环境说明见 [setup.md](setup.md)。

## 示例与约定

- “把笔记本向右移动一步，沿桌面旋转 90 度，检查碰撞，最后保存场景。”
- “只显示笔记本，隐藏包围盒，然后聚焦它。”
- “关闭贴桌约束，把笔记本沿 Z 轴抬高 0.1 个重建单位。”
- 已加载独立的 Laptop 和 Cup 时：“把 Laptop 向右移动 0.1，让 Cup 靠近 Laptop，间隔 0.03，然后保存场景。”
- “把笔记本恢复到原始位置。”

右/前/上分别约定为世界 +X/+Y/+Z，和相机视角无关。没有给距离时使用现有 Translation step；
没有米制标定，因此“移动 10 厘米”应先询问换算比例。贴桌开关打开时沿用原有平面约束；
要求抬高或倾斜时，计划需明确关闭它。移动是增量，旋转使用世界轴或桌面法向，遵循右手规则。

LLM 只生成结构化工具计划。确定性的执行器先验证整条计划、模拟变换和碰撞检查，再逐步执行。
模型给出的“计划说明”不作为执行成功依据；界面展示实际位移、检查结果和保存路径。
生成计划期间如果用户修改场景，计划作废，需要重新提交。最近三轮成功交互用于理解“再往右一点”等指令。

## 物体查找与多物体编辑

`find_object` 在启动参数 `--object LABEL=OBJECT_NPZ` 加载的候选中选择，不调用新的分割或重建算法。
模型可将中文物体名对应到候选标签；候选不明确时应先询问用户，在明确前返回空计划。
找不到水杯时不能生成水杯：先用项目原有 CLIP 查询和候选准备流程生成选择，
再用多个 `--object` 参数启动同一个查看器。所有选择必须来自同一份 `--scene`，
并符合查看器已有的原始高斯索引和数组一致性要求。详见 [semantic_object_pipeline.md](semantic_object_pipeline.md)。
默认演示只加载笔记本，因此不能直接执行水杯相关操作。

每个候选拥有自己的 `ObjectState`。切换物体保留已有变换，其余已编辑物体显示在背景中；
Reset object 只复位当前物体。这是为多物体语言指令所需的行为调整。
同一物体的多个重叠分割候选不能同时编辑；先复位前一个候选，再编辑另一个。

`place_near` 保留旋转，沿两个物体包围盒中心的连线放到目标外侧；贴桌模式下使用桌面内方向。
`check_collision` 用变换后高斯中心的世界轴对齐包围盒检测已加载且索引不重叠的候选。
移动和旋转也自动检查目标位置，包围盒相交会拒绝整条计划。它是保守的粗略检测，
不是网格碰撞、轨迹规划或物理仿真；不检查未分割的背景、桌面边界和移动途中的碰撞。
复位允许返回原始位置。

## 工具与保存

支持 `find_object`、`move_object`、`rotate_object`、`place_near`、`check_collision`、
`reset_object`、`set_display`、`focus_object`、`save_scene`。不执行模型生成的 Python 或 shell 命令。
单条指令最多 24 步。显示开关只使用场景中现有、可用的控件。

`save_scene` 在原有 `--export-dir`（默认 `outputs/object_edits/`）写入：

- `scene_<时间>_edited.npz`：完整原场景与全部独立物体的刚体编辑。更新中心和四元数，保持颜色、尺度和不透明度。
- 同名 `.json`：原场景路径、已编辑物体的位姿、单位和背景说明。

它保存高斯场景数据，不保存训练 checkpoint、网页显示开关或推断桌面层。
原模型与输入 NPZ 不覆盖。原有 Export object and pose 仍然只导出选中物体。
网络失败、计划无法解析、未知候选和碰撞拒绝都不会执行计划；执行中的文件写入错误会显示为未完成，已执行的几何操作仍可用 Reset object 复位。

## 验证

```bash
python -m pytest tests/test_language_agent.py tests/test_pose_viewer.py -q
```

测试包含多物体变换、包围盒摆放与碰撞拒绝、原始数据保留、整场景导出、平面约束、
无效计划不执行、重叠候选拒绝，以及 DeepSeek Flash 非思考请求格式。
网络返回在测试中替代为固定响应，真实模型调用需要有效密钥。

API 参数依据：[DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)、
[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)。
