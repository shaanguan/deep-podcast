# 🎙️ Deep Podcast

基于 ChatTTS 的多角色 AI 配音生成器，支持无限角色、自动分配声纹、Web UI 界面。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ChatTTS](https://img.shields.io/badge/ChatTTS-0.2.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ 功能特性

- 🎭 **无限角色支持** - 自动识别发言人，动态分配不同声纹
- ⚙️ **配置化管理** - 通过 YAML 配置文件定义角色声音和演技参数
- 🔄 **断点续传** - 分段缓存音频，支持中断恢复
- 📝 **文本预处理** - 数字转汉字、保留 ChatTTS 标签、移除时间戳
- 🔊 **音频优化** - 响度归一化、交叉淡入淡出、底噪填充
- 💻 **硬件自适应** - 自动检测 CUDA/MPS/CPU
- 🌐 **Web UI** - 美观易用的 Gradio 界面
- 📁 **多格式支持** - 支持导入 Word/JSON/Markdown/TXT 文件

## 📸 界面预览

Web UI 提供以下功能：
- 文本输入 / 文件导入
- 角色声纹配置
- 音频参数调节
- 实时解析预览
- 音频播放和下载

## 🖥️ 系统要求

### 硬件要求

| 设备 | 推荐配置 | 生成速度 |
|------|----------|----------|
| NVIDIA GPU | 4GB+ 显存 | ~10x 实时 |
| Apple Silicon | M1/M2/M3 | ~5x 实时 |
| CPU | 8GB+ 内存 | ~0.5x 实时 |

### 软件依赖

- Python 3.9 - 3.11（不支持 3.12+）
- FFmpeg（pydub 底层依赖）

#### FFmpeg 安装

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/deep-podcast.git
cd deep-podcast
```

### 2. 创建虚拟环境

```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 启动 Web UI

```bash
python app.py
```

访问 http://127.0.0.1:7860

### 5. 或使用命令行

```bash
# 基本用法
python main.py -i data/input/transcript.txt -o data/output/podcast.wav

# 空跑模式（只解析，不生成）
python main.py --dry-run

# 指定范围生成
python main.py --range 5-10
```

## 📖 使用说明

### 文本格式

```
发言人1 大家好，欢迎收听本期节目！

发言人2 你好，很高兴来到这里。

发言人1 今天我们来聊聊人工智能的话题。
```

### 支持的导入格式

| 格式 | 说明 |
|------|------|
| `.txt` | 纯文本，`发言人N 内容` 格式 |
| `.docx` | Word 文档，自动提取段落 |
| `.json` | JSON 数组 `[{"speaker": "1", "text": "..."}]` |
| `.md` | Markdown，支持 `## 发言人N` 或 `**发言人N**` 格式 |

### 配置文件 (config.yaml)

```yaml
# 固定角色配置
roles:
  "1":
    seed: 3333            # 声纹种子
    prompt: "[speed_5]"   # 语速
    desc: "主持人"

# 自动种子策略
auto_seed:
  base: 5000
  step: 337

# 音频配置
audio:
  sample_rate: 24000
  pause_duration: 0.5
  normalize: true
```

### 演技参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| `oral` | 0-9 | 口语化程度 |
| `laugh` | 0-9 | 笑声概率 |
| `break` | 0-9 | 停顿概率 |
| `speed` | 1-9 | 语速 |

## 📁 项目结构

```
Deep Podcast/
├── app.py                  # Web UI 入口
├── main.py                 # CLI 入口
├── config.yaml             # 配置文件
├── requirements.txt        # Python 依赖
├── src/
│   ├── parser.py           # 文档解析器
│   ├── role_manager.py     # 角色管理器
│   ├── generator.py        # 音频生成器
│   ├── text_normalizer.py  # 文本预处理
│   └── utils.py            # 工具函数
├── data/
│   ├── input/              # 输入文件
│   └── output/             # 输出音频
├── temp/                   # 缓存目录
└── assets/                 # 静态资源
```

## ❓ 常见问题

### Q: 模型下载失败？

使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python main.py
```

### Q: CUDA 报错 / 显存不足？

1. 减少 `config.yaml` 中的 `max_chunk_length`
2. 使用 `--range` 参数分段生成

### Q: pydub 报错？

确保已安装 FFmpeg：
```bash
ffmpeg -version
```

### Q: 声音不稳定？

1. 降低 `temperature` 参数（越小越稳定）
2. 手动指定固定 seed

## 📄 许可证

MIT License

## 🙏 致谢

- [ChatTTS](https://github.com/2noise/ChatTTS) - 强大的中文语音合成模型
- [Gradio](https://gradio.app/) - Web UI 框架
- [cn2an](https://github.com/Ailln/cn2an) - 中文数字转换库
