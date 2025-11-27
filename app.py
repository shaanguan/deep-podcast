#!/usr/bin/env python3
"""
Deep Podcast - Web UI
基于 Gradio 的可视化界面
"""

import os
import sys
import json
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import yaml
import gradio as gr

# 用于解析 Word 文档
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import parse_transcript, DialogueItem
from src.role_manager import RoleManager
from src.generator import AudioGenerator
from src.utils import print_info, print_success, check_hardware


# ============================================================
# 全局变量
# ============================================================
GENERATOR: Optional[AudioGenerator] = None
ROLE_MANAGER: Optional[RoleManager] = None
CONFIG: dict = {}
MODEL_LOADED = False


# ============================================================
# 工具函数
# ============================================================

def load_config() -> dict:
    """加载配置文件"""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict):
    """保存配置文件"""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def init_model():
    """初始化模型（延迟加载）"""
    global GENERATOR, ROLE_MANAGER, CONFIG, MODEL_LOADED
    
    if MODEL_LOADED:
        return "✅ 模型已加载"
    
    try:
        CONFIG = load_config()
        ROLE_MANAGER = RoleManager(CONFIG)
        GENERATOR = AudioGenerator(CONFIG, ROLE_MANAGER)
        
        temp_dir = str(PROJECT_ROOT / "temp")
        assets_dir = str(PROJECT_ROOT / "assets")
        
        if not GENERATOR.initialize(temp_dir=temp_dir, assets_dir=assets_dir):
            return "❌ 模型加载失败"
        
        MODEL_LOADED = True
        return "✅ 模型加载成功！"
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"


def parse_uploaded_file(file_path: str) -> str:
    """
    解析上传的文件，支持 Word、JSON、Markdown、TXT 格式
    
    Args:
        file_path: 上传文件的路径
        
    Returns:
        解析后的文本内容
    """
    if not file_path:
        return ""
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        # Word 文档 (.docx)
        if file_ext == '.docx':
            if not HAS_DOCX:
                return "❌ 请安装 python-docx: pip install python-docx"
            
            doc = Document(file_path)
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            return '\n\n'.join(paragraphs)
        
        # JSON 文件 (.json)
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持多种 JSON 格式
            lines = []
            
            # 格式1: {"dialogues": [{"speaker": "1", "text": "..."}]}
            if isinstance(data, dict) and 'dialogues' in data:
                for item in data['dialogues']:
                    speaker = item.get('speaker', item.get('speaker_id', item.get('id', '1')))
                    text = item.get('text', item.get('content', ''))
                    lines.append(f"发言人{speaker} {text}")
            
            # 格式2: [{"speaker": "1", "text": "..."}]
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        speaker = item.get('speaker', item.get('speaker_id', item.get('id', '1')))
                        text = item.get('text', item.get('content', ''))
                        lines.append(f"发言人{speaker} {text}")
                    elif isinstance(item, str):
                        lines.append(item)
            
            # 格式3: {"1": ["text1", "text2"], "2": ["text1"]}
            elif isinstance(data, dict):
                for speaker, texts in data.items():
                    if isinstance(texts, list):
                        for text in texts:
                            lines.append(f"发言人{speaker} {text}")
                    elif isinstance(texts, str):
                        lines.append(f"发言人{speaker} {texts}")
            
            return '\n\n'.join(lines)
        
        # Markdown 文件 (.md)
        elif file_ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 处理 Markdown 格式
            lines = []
            current_speaker = None
            current_text = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 检测标题格式的发言人标记: ## 发言人1 或 ### Speaker 1
                header_match = re.match(r'^#{1,6}\s*(发言人|Speaker|说话人)\s*(\d+)\s*[:：]?\s*(.*)$', line, re.IGNORECASE)
                if header_match:
                    # 保存上一个发言人的内容
                    if current_speaker and current_text:
                        lines.append(f"发言人{current_speaker} {''.join(current_text)}")
                    current_speaker = header_match.group(2)
                    current_text = [header_match.group(3)] if header_match.group(3) else []
                    continue
                
                # 检测粗体格式: **发言人1**: 内容
                bold_match = re.match(r'^\*\*(发言人|Speaker|说话人)\s*(\d+)\*\*\s*[:：]?\s*(.*)$', line, re.IGNORECASE)
                if bold_match:
                    if current_speaker and current_text:
                        lines.append(f"发言人{current_speaker} {''.join(current_text)}")
                    current_speaker = bold_match.group(2)
                    current_text = [bold_match.group(3)] if bold_match.group(3) else []
                    continue
                
                # 检测普通格式: 发言人1: 内容
                speaker_match = re.match(r'^(发言人|Speaker|说话人)\s*(\d+)\s*[:：]?\s*(.*)$', line, re.IGNORECASE)
                if speaker_match:
                    if current_speaker and current_text:
                        lines.append(f"发言人{current_speaker} {''.join(current_text)}")
                    current_speaker = speaker_match.group(2)
                    current_text = [speaker_match.group(3)] if speaker_match.group(3) else []
                    continue
                
                # 跳过 Markdown 格式符号
                if line.startswith(('---', '***', '===', '```')):
                    continue
                
                # 移除 Markdown 链接和图片
                line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
                line = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', line)
                
                # 移除其他 Markdown 格式
                line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)  # 粗体
                line = re.sub(r'\*([^*]+)\*', r'\1', line)      # 斜体
                line = re.sub(r'`([^`]+)`', r'\1', line)        # 代码
                
                if current_speaker and line:
                    current_text.append(line)
            
            # 保存最后一个发言人
            if current_speaker and current_text:
                lines.append(f"发言人{current_speaker} {''.join(current_text)}")
            
            # 如果没有识别到发言人格式，返回原始内容
            if not lines:
                return content
            
            return '\n\n'.join(lines)
        
        # 纯文本文件 (.txt)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        return f"❌ 文件解析失败: {str(e)}"


def parse_text_to_dialogues(text: str) -> Tuple[str, list]:
    """解析文本为对话列表"""
    if not text.strip():
        return "⚠️ 请输入文本内容", []
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name
    
    try:
        # 重新加载配置（可能已更改）
        config = load_config()
        dialogues = parse_transcript(
            temp_path,
            normalize_text=True,
            convert_numbers=config.get('text', {}).get('convert_numbers', True)
        )
        
        if not dialogues:
            return "⚠️ 未解析到有效对话，请检查格式", []
        
        # 统计信息
        speaker_stats = {}  # {speaker_id: {"count": 0, "chars": 0}}
        total_chars = 0
        
        for d in dialogues:
            if d.speaker_id not in speaker_stats:
                speaker_stats[d.speaker_id] = {"count": 0, "chars": 0}
            speaker_stats[d.speaker_id]["count"] += 1
            speaker_stats[d.speaker_id]["chars"] += len(d.text)
            total_chars += len(d.text)
        
        # 预估时长（粗略估计：每字约 0.3 秒）
        estimated_seconds = total_chars * 0.3
        est_minutes = int(estimated_seconds // 60)
        est_seconds = int(estimated_seconds % 60)
        
        # 生成统计摘要
        preview_lines = []
        preview_lines.append("=" * 40)
        preview_lines.append("📊 统计信息")
        preview_lines.append("=" * 40)
        preview_lines.append(f"👥 角色数量: {len(speaker_stats)} 位")
        preview_lines.append(f"💬 对话片段: {len(dialogues)} 段")
        preview_lines.append(f"📝 总字数: {total_chars} 字")
        preview_lines.append(f"⏱️ 预估时长: 约 {est_minutes}分{est_seconds}秒")
        preview_lines.append("")
        preview_lines.append("📋 各角色统计:")
        
        for spk_id in sorted(speaker_stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            stats = speaker_stats[spk_id]
            preview_lines.append(f"   发言人{spk_id}: {stats['count']} 段, {stats['chars']} 字")
        
        preview_lines.append("")
        preview_lines.append("=" * 40)
        preview_lines.append("📜 对话预览 (前10条)")
        preview_lines.append("=" * 40)
        
        # 只显示前10条对话预览
        for d in dialogues[:10]:
            text_preview = d.text[:40] + "..." if len(d.text) > 40 else d.text
            preview_lines.append(f"[{d.index + 1:02d}] 发言人{d.speaker_id}: {text_preview}")
        
        if len(dialogues) > 10:
            preview_lines.append(f"... 还有 {len(dialogues) - 10} 条对话")
        
        preview = "\n".join(preview_lines)
        
        return preview, dialogues
    finally:
        os.unlink(temp_path)


# ============================================================
# Gradio 回调函数
# ============================================================

def on_load_model():
    """加载模型按钮回调"""
    return init_model()


def on_parse_text(text: str):
    """解析文本按钮回调"""
    preview, _ = parse_text_to_dialogues(text)
    return preview


def on_file_upload(file):
    """文件上传回调"""
    if file is None:
        return "", "⚠️ 请选择文件"
    
    # Gradio 上传的文件是临时文件路径
    file_path = file.name if hasattr(file, 'name') else file
    
    # 解析文件
    text = parse_uploaded_file(file_path)
    
    if text.startswith("❌"):
        return "", text
    
    # 预览解析结果
    preview, _ = parse_text_to_dialogues(text)
    
    return text, preview


def on_generate(
    text: str,
    host_seed: int,
    host_speed: int,
    guest_seed: int,
    guest_speed: int,
    auto_seed_base: int,
    pause_duration: float,
    normalize: bool,
    temperature: float
):
    """生成音频按钮回调"""
    global GENERATOR, ROLE_MANAGER, CONFIG, MODEL_LOADED
    
    if not MODEL_LOADED:
        return None, "❌ 请先加载模型"
    
    if not text.strip():
        return None, "❌ 请输入文本内容"
    
    # 更新配置
    CONFIG['roles'] = {
        "1": {
            "seed": host_seed,
            "prompt": f"[speed_{host_speed}]",
            "desc": "主持人",
            "refine_override": {"oral": 0, "laugh": 0, "break": 3}
        },
        "2": {
            "seed": guest_seed,
            "prompt": f"[speed_{guest_speed}]",
            "desc": "嘉宾",
            "refine_override": {"oral": 2, "laugh": 1, "break": 4}
        }
    }
    CONFIG['auto_seed']['base'] = auto_seed_base
    CONFIG['audio']['pause_duration'] = pause_duration
    CONFIG['audio']['normalize'] = normalize
    CONFIG['style']['temperature'] = temperature
    
    # 重新初始化角色管理器
    ROLE_MANAGER = RoleManager(CONFIG)
    GENERATOR.role_manager = ROLE_MANAGER
    GENERATOR.pause_duration = pause_duration
    GENERATOR.do_normalize = normalize
    GENERATOR.temperature = temperature
    
    # 解析文本
    _, dialogues = parse_text_to_dialogues(text)
    if not dialogues:
        return None, "❌ 解析失败"
    
    # 生成音频
    output_path = str(PROJECT_ROOT / "data" / "output" / "podcast_ui.wav")
    
    try:
        success = GENERATOR.generate(
            dialogues,
            output_path,
            resume=False,
            force=True
        )
        
        if success and os.path.exists(output_path):
            # 计算时长
            import soundfile as sf
            data, sr = sf.read(output_path)
            duration = len(data) / sr
            minutes = int(duration // 60)
            seconds = duration % 60
            
            return output_path, f"✅ 生成成功！时长: {minutes}分{seconds:.1f}秒"
        else:
            return None, "❌ 生成失败"
            
    except Exception as e:
        return None, f"❌ 生成出错: {str(e)}"


def on_load_sample():
    """加载示例文本"""
    sample_path = PROJECT_ROOT / "data" / "input" / "transcript.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# ============================================================
# Gradio UI
# ============================================================

def create_ui():
    """创建 Gradio 界面"""
    
    # 自定义 CSS
    custom_css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 1.5em;
    }
    """
    
    with gr.Blocks(
        title="Deep Podcast - AI 配音生成器",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        )
    ) as app:
        
        # 标题
        gr.HTML("""
            <div class="main-title">🎙️ Deep Podcast</div>
            <div class="subtitle">基于 ChatTTS 的多角色 AI 配音生成器</div>
        """)
        
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### 📝 文本输入")
                
                # 文件上传
                with gr.Accordion("📁 导入文件 (Word/JSON/Markdown/TXT)", open=False):
                    file_upload = gr.File(
                        label="选择文件",
                        file_types=[".docx", ".json", ".md", ".txt"],
                        file_count="single"
                    )
                    gr.Markdown("""
                    **支持格式：**
                    - `.docx` - Word 文档
                    - `.json` - JSON 格式 `[{"speaker": "1", "text": "..."}]`
                    - `.md` - Markdown（支持 `## 发言人1` 或 `**发言人1**` 格式）
                    - `.txt` - 纯文本
                    """)
                
                text_input = gr.Textbox(
                    label="对话文本",
                    placeholder="发言人1 大家好，欢迎收听...\n发言人2 你好...",
                    lines=15,
                    max_lines=30
                )
                
                with gr.Row():
                    load_sample_btn = gr.Button("📄 加载示例", size="sm")
                    parse_btn = gr.Button("🔍 解析预览", size="sm", variant="secondary")
                
                parse_output = gr.Textbox(
                    label="解析结果",
                    lines=8,
                    interactive=False
                )
            
            # 右侧：参数配置
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 参数配置")
                
                with gr.Tabs():
                    # 角色配置
                    with gr.TabItem("👤 角色设置"):
                        gr.Markdown("**主持人 (发言人1)**")
                        with gr.Row():
                            host_seed = gr.Slider(
                                minimum=1, maximum=10000, value=3333, step=1,
                                label="声纹种子", info="决定音色"
                            )
                            host_speed = gr.Slider(
                                minimum=1, maximum=9, value=5, step=1,
                                label="语速", info="1慢-9快"
                            )
                        
                        gr.Markdown("**嘉宾 (发言人2)**")
                        with gr.Row():
                            guest_seed = gr.Slider(
                                minimum=1, maximum=10000, value=5674, step=1,
                                label="声纹种子", info="决定音色"
                            )
                            guest_speed = gr.Slider(
                                minimum=1, maximum=9, value=5, step=1,
                                label="语速", info="1慢-9快"
                            )
                        
                        gr.Markdown("**其他角色 (自动分配)**")
                        auto_seed_base = gr.Slider(
                            minimum=1000, maximum=9000, value=5000, step=100,
                            label="自动种子基数", info="发言人N的种子 = 基数 + N*337"
                        )
                    
                    # 音频配置
                    with gr.TabItem("🔊 音频设置"):
                        pause_duration = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                            label="段落停顿 (秒)"
                        )
                        normalize = gr.Checkbox(
                            value=True,
                            label="音频归一化",
                            info="统一响度"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                            label="随机性",
                            info="越低越稳定，越高越多变"
                        )
        
        # 底部：操作区
        gr.Markdown("---")
        
        with gr.Row():
            load_model_btn = gr.Button("🚀 加载模型", variant="secondary", scale=1)
            generate_btn = gr.Button("🎵 生成音频", variant="primary", scale=2)
        
        model_status = gr.Textbox(
            label="模型状态",
            value="⏳ 模型未加载，请点击「加载模型」",
            interactive=False
        )
        
        gr.Markdown("### 🎧 生成结果")
        
        with gr.Row():
            audio_output = gr.Audio(
                label="生成的音频",
                type="filepath",
                interactive=False
            )
            generation_status = gr.Textbox(
                label="生成状态",
                interactive=False,
                lines=3
            )
        
        # 绑定事件
        file_upload.change(
            fn=on_file_upload,
            inputs=file_upload,
            outputs=[text_input, parse_output]
        )
        
        load_sample_btn.click(
            fn=on_load_sample,
            outputs=text_input
        )
        
        parse_btn.click(
            fn=on_parse_text,
            inputs=text_input,
            outputs=parse_output
        )
        
        load_model_btn.click(
            fn=on_load_model,
            outputs=model_status
        )
        
        generate_btn.click(
            fn=on_generate,
            inputs=[
                text_input,
                host_seed, host_speed,
                guest_seed, guest_speed,
                auto_seed_base,
                pause_duration, normalize, temperature
            ],
            outputs=[audio_output, generation_status]
        )
        
        # 页脚
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #888; font-size: 0.9em;">
            Deep Podcast v1.0 | Powered by ChatTTS | 
            <a href="https://github.com/2noise/ChatTTS" target="_blank">GitHub</a>
        </div>
        """)
    
    return app


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 检测硬件
    print("=" * 50)
    print("  Deep Podcast - Web UI")
    print("=" * 50)
    check_hardware()
    print()
    
    # 创建并启动 UI
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

