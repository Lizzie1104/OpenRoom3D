# OpenRoom3D：3D 视觉课程项目

用一段桌面视频重建 3D 高斯场景，用文字查询其中的物体，再把选中的物体抓出来移动。

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
