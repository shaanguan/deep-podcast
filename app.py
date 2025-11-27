#!/usr/bin/env python3
"""
Deep Podcast - Premium Web UI
High-fidelity modern interface with glassmorphism design
"""

import os
import sys
import json
import re
import time
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple, Generator
from dataclasses import dataclass

import yaml
import numpy as np
import gradio as gr

# 用于解析 Word 文档
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import parse_transcript, DialogueItem
from src.role_manager import RoleManager
from src.generator import AudioGenerator
from src.utils import print_info, print_success, check_hardware


# ============================================================
# 全局状态
# ============================================================
@dataclass
class AppState:
    generator: Optional[AudioGenerator] = None
    role_manager: Optional[RoleManager] = None
    config: dict = None
    model_loaded: bool = False
    is_generating: bool = False
    
APP_STATE = AppState()


# ============================================================
# 配置管理
# ============================================================

def load_config() -> dict:
    """加载配置文件"""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# 模型管理
# ============================================================

def ensure_model_loaded(progress_callback=None) -> Tuple[bool, str]:
    """确保模型已加载"""
    global APP_STATE
    
    if APP_STATE.model_loaded and APP_STATE.generator is not None:
        return True, "模型就绪"
    
    try:
        if progress_callback:
            progress_callback(0.1, "加载配置...")
        
        APP_STATE.config = load_config()
        APP_STATE.role_manager = RoleManager(APP_STATE.config)
        APP_STATE.generator = AudioGenerator(APP_STATE.config, APP_STATE.role_manager)
        
        if progress_callback:
            progress_callback(0.2, "加载 ChatTTS 模型...")
        
        temp_dir = str(PROJECT_ROOT / "temp")
        assets_dir = str(PROJECT_ROOT / "assets")
        
        if not APP_STATE.generator.initialize(temp_dir=temp_dir, assets_dir=assets_dir):
            return False, "模型加载失败"
        
        APP_STATE.model_loaded = True
        return True, "模型加载成功"
        
    except Exception as e:
        return False, f"加载失败: {str(e)}"


# ============================================================
# 文件解析
# ============================================================

def parse_uploaded_file(file_path: str) -> str:
    """解析上传的文件"""
    if not file_path:
        return ""
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.docx':
            if not HAS_DOCX:
                return "❌ 请安装 python-docx"
            doc = Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return '\n\n'.join(paragraphs)
        
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            lines = []
            if isinstance(data, dict) and 'dialogues' in data:
                for item in data['dialogues']:
                    speaker = item.get('speaker', '1')
                    text = item.get('text', '')
                    lines.append(f"发言人{speaker} {text}")
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        speaker = item.get('speaker', '1')
                        text = item.get('text', '')
                        lines.append(f"发言人{speaker} {text}")
            return '\n\n'.join(lines)
        
        elif file_ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        return f"❌ 解析失败: {str(e)}"


def parse_text_to_dialogues(text: str) -> Tuple[str, list]:
    """解析文本为对话列表"""
    if not text.strip():
        return "请输入脚本内容", []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name
    
    try:
        config = load_config()
        dialogues = parse_transcript(
            temp_path,
            normalize_text=True,
            convert_numbers=config.get('text', {}).get('convert_numbers', True)
        )
        
        if not dialogues:
            return "未解析到有效对话", []
        
        # 统计
        speaker_stats = {}
        total_chars = 0
        
        for d in dialogues:
            if d.speaker_id not in speaker_stats:
                speaker_stats[d.speaker_id] = {"count": 0, "chars": 0}
            speaker_stats[d.speaker_id]["count"] += 1
            speaker_stats[d.speaker_id]["chars"] += len(d.text)
            total_chars += len(d.text)
        
        # 预估时长
        estimated_seconds = total_chars * 0.3
        est_minutes = int(estimated_seconds // 60)
        est_seconds = int(estimated_seconds % 60)
        
        # 简洁的统计信息
        stats = f"✓ {len(dialogues)} 段对话 · {len(speaker_stats)} 位角色 · {total_chars} 字 · 约 {est_minutes}:{est_seconds:02d}"
        
        return stats, dialogues
        
    finally:
        os.unlink(temp_path)


# ============================================================
# 核心生成逻辑
# ============================================================

def generate_audio(
    text: str,
    host_seed: int,
    host_speed: int,
    host_emotion: int,
    guest_seed: int,
    guest_speed: int,
    guest_emotion: int,
    pause_duration: float,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], str]:
    """生成音频"""
    global APP_STATE
    
    if APP_STATE.is_generating:
        return None, "⏳ 生成中，请稍候..."
    
    APP_STATE.is_generating = True
    start_time = time.time()
    
    try:
        # Step 1: 加载模型
        progress(0.05, desc="Initializing...")
        
        success, msg = ensure_model_loaded(
            lambda p, s: progress(p * 0.25, desc=s)
        )
        
        if not success:
            return None, f"❌ {msg}"
        
        # Step 2: 配置
        progress(0.25, desc="Configuring voices...")
        
        APP_STATE.config['roles'] = {
            "1": {
                "seed": host_seed,
                "prompt": f"[speed_{host_speed}]",
                "desc": "Host",
                "refine_override": {
                    "oral": min(host_emotion, 5),
                    "laugh": max(0, host_emotion - 5),
                    "break": 3
                }
            },
            "2": {
                "seed": guest_seed,
                "prompt": f"[speed_{guest_speed}]",
                "desc": "Guest",
                "refine_override": {
                    "oral": min(guest_emotion, 5),
                    "laugh": max(0, guest_emotion - 5),
                    "break": 4
                }
            }
        }
        APP_STATE.config['audio']['pause_duration'] = pause_duration
        
        APP_STATE.role_manager = RoleManager(APP_STATE.config)
        APP_STATE.generator.role_manager = APP_STATE.role_manager
        APP_STATE.generator.pause_duration = pause_duration
        
        # Step 3: 解析
        progress(0.28, desc="Parsing script...")
        _, dialogues = parse_text_to_dialogues(text)
        
        if not dialogues:
            return None, "❌ 解析失败"
        
        total_dialogues = len(dialogues)
        
        # Step 4: 生成
        progress(0.30, desc=f"Generating {total_dialogues} segments...")
        
        output_path = str(PROJECT_ROOT / "data" / "output" / "podcast_ui.wav")
        
        # 后台生成
        generation_complete = threading.Event()
        generation_result = {"success": False, "error": None}
        
        def run_generation():
            try:
                result = APP_STATE.generator.generate(
                    dialogues, output_path, resume=False, force=True
                )
                generation_result["success"] = result
            except Exception as e:
                generation_result["error"] = str(e)
            finally:
                generation_complete.set()
        
        gen_thread = threading.Thread(target=run_generation)
        gen_thread.start()
        
        # 进度更新
        while not generation_complete.is_set():
            temp_dir = PROJECT_ROOT / "temp"
            if temp_dir.exists():
                done = len(list(temp_dir.glob("*.wav")))
                pct = 0.30 + (done / max(total_dialogues, 1)) * 0.60
                elapsed = time.time() - start_time
                if done > 0:
                    eta = int((elapsed / done) * (total_dialogues - done))
                    progress(min(pct, 0.90), desc=f"Generating... {done}/{total_dialogues} (ETA: {eta}s)")
                else:
                    progress(0.35, desc="Generating...")
            time.sleep(0.5)
        
        gen_thread.join()
        
        if generation_result["error"]:
            return None, f"❌ {generation_result['error']}"
        
        if not generation_result["success"] or not os.path.exists(output_path):
            return None, "❌ 生成失败"
        
        progress(0.95, desc="Finalizing...")
        
        # 计算结果
        data, sr = sf.read(output_path)
        duration = len(data) / sr
        elapsed_total = time.time() - start_time
        
        progress(1.0, desc="Complete!")
        
        result_msg = f"✓ {int(duration//60)}:{int(duration%60):02d} 音频 · 耗时 {int(elapsed_total)}s"
        
        return output_path, result_msg
        
    except Exception as e:
        return None, f"❌ {str(e)}"
    
    finally:
        APP_STATE.is_generating = False


# ============================================================
# UI 回调
# ============================================================

def on_file_upload(file):
    if file is None:
        return "", ""
    file_path = file.name if hasattr(file, 'name') else file
    text = parse_uploaded_file(file_path)
    if text.startswith("❌"):
        return "", text
    stats, _ = parse_text_to_dialogues(text)
    return text, stats


def on_text_change(text):
    if not text.strip():
        return ""
    stats, _ = parse_text_to_dialogues(text)
    return stats


def on_load_sample():
    sample_path = PROJECT_ROOT / "data" / "input" / "transcript.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            text = f.read()
        stats, _ = parse_text_to_dialogues(text)
        return text, stats
    return "", ""


# ============================================================
# Premium UI
# ============================================================

def create_ui():
    """创建高保真 UI"""
    
    css = """
    /* ===== 全局样式 ===== */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        background: linear-gradient(180deg, #f8f9fc 0%, #eef1f8 100%) !important;
        min-height: 100vh;
    }
    
    /* ===== 头部 ===== */
    .header-section {
        text-align: center;
        padding: 40px 20px 30px;
        margin-bottom: 20px;
    }
    
    .logo-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    
    .logo-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    }
    
    .app-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        color: #6b7280;
        font-size: 15px;
        font-weight: 400;
    }
    
    /* ===== 玻璃卡片 ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 20px !important;
        box-shadow: 
            0 4px 24px rgba(102, 126, 234, 0.08),
            0 1px 2px rgba(0, 0, 0, 0.04) !important;
        padding: 24px !important;
    }
    
    /* ===== 面板标题 ===== */
    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .panel-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .panel-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
    }
    
    /* ===== 角色卡片 ===== */
    .character-card {
        background: linear-gradient(135deg, #fafbff 0%, #f3f4f8 100%);
        border: 1px solid #e8ebf4;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    
    .character-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    
    .character-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    
    .avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .avatar-host {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .avatar-guest {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .character-name {
        font-size: 16px;
        font-weight: 600;
        color: #374151;
    }
    
    .character-role {
        font-size: 13px;
        color: #9ca3af;
    }
    
    /* ===== 滑块美化 ===== */
    input[type="range"] {
        -webkit-appearance: none;
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        cursor: pointer;
    }
    
    /* ===== 上传区域 ===== */
    .upload-zone {
        border: 2px dashed #d1d5db !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #fafbff 0%, #f8f9fc 100%) !important;
        padding: 24px !important;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #667eea !important;
        background: linear-gradient(135deg, #f5f7ff 0%, #f0f3ff 100%) !important;
    }
    
    /* ===== 文本编辑器 ===== */
    textarea {
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
        background: #fafbfc !important;
        transition: all 0.2s ease !important;
    }
    
    textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* ===== 生成按钮 ===== */
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 48px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 
            0 8px 32px rgba(102, 126, 234, 0.35),
            0 0 0 0 rgba(102, 126, 234, 0.5) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            0 12px 40px rgba(102, 126, 234, 0.45),
            0 0 30px rgba(102, 126, 234, 0.3) !important;
    }
    
    .generate-btn:active {
        transform: translateY(0) !important;
    }
    
    /* ===== 状态标签 ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }
    
    .status-pending {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
    }
    
    /* ===== 音频播放器 ===== */
    .audio-player {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-top: 20px;
    }
    
    .audio-player audio {
        width: 100%;
    }
    
    /* ===== 统计信息 ===== */
    .stats-display {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 13px;
        color: #166534;
        font-weight: 500;
    }
    
    /* ===== 响应式 ===== */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
        }
        .app-title {
            font-size: 24px;
        }
    }
    """
    
    with gr.Blocks(
        title="Deep Podcast - AI Voice Studio",
        css=css,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        )
    ) as app:
        
        # ===== 头部 =====
        gr.HTML("""
        <div class="header-section">
            <div class="logo-title">
                <div class="logo-icon">🎙️</div>
                <span class="app-title">Deep Podcast</span>
            </div>
            <p class="app-subtitle">AI-Powered Multi-Voice Podcast Generator</p>
        </div>
        """)
        
        # ===== 主内容 =====
        with gr.Row(equal_height=True):
            
            # ===== 左栏: 脚本编辑器 =====
            with gr.Column(scale=6):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">📝</div>
                    <span class="panel-title">Script Editor</span>
                </div>
                """)
                
                # 文件上传
                file_upload = gr.File(
                    label="",
                    file_types=[".docx", ".json", ".md", ".txt"],
                    file_count="single",
                    elem_classes=["upload-zone"]
                )
                gr.HTML("""
                <div style="text-align: center; margin: -10px 0 16px; color: #9ca3af; font-size: 13px;">
                    📄 Drop Word, JSON, Markdown or Text files here
                </div>
                """)
                
                # 文本编辑器
                text_input = gr.Textbox(
                    label="",
                    placeholder="发言人1 大家好，欢迎收听本期播客！\n\n发言人2 今天我们来聊聊人工智能...",
                    lines=14,
                    max_lines=20,
                    show_label=False
                )
                
                # 统计信息
                stats_display = gr.HTML(
                    value='<div class="stats-display">📊 等待输入脚本...</div>'
                )
                
                with gr.Row():
                    load_sample_btn = gr.Button("📄 Load Sample", size="sm", variant="secondary")
            
            # ===== 右栏: 角色设置 =====
            with gr.Column(scale=5):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">🎭</div>
                    <span class="panel-title">Cast Studio</span>
                </div>
                """)
                
                # 主持人卡片
                gr.HTML("""
                <div class="character-card">
                    <div class="character-header">
                        <div class="avatar avatar-host">👩</div>
                        <div>
                            <div class="character-name">Host</div>
                            <div class="character-role">发言人 1 · 主持人</div>
                        </div>
                    </div>
                </div>
                """)
                
                with gr.Group():
                    host_seed = gr.Slider(
                        minimum=1, maximum=10000, value=3333, step=1,
                        label="🎤 Voice Seed",
                        info="控制音色特征"
                    )
                    with gr.Row():
                        host_speed = gr.Slider(
                            minimum=1, maximum=9, value=5, step=1,
                            label="⚡ Speed"
                        )
                        host_emotion = gr.Slider(
                            minimum=0, maximum=9, value=3, step=1,
                            label="💫 Emotion"
                        )
                
                gr.HTML("<div style='height: 16px'></div>")
                
                # 嘉宾卡片
                gr.HTML("""
                <div class="character-card">
                    <div class="character-header">
                        <div class="avatar avatar-guest">👨</div>
                        <div>
                            <div class="character-name">Guest</div>
                            <div class="character-role">发言人 2 · 嘉宾</div>
                        </div>
                    </div>
                </div>
                """)
                
                with gr.Group():
                    guest_seed = gr.Slider(
                        minimum=1, maximum=10000, value=5674, step=1,
                        label="🎤 Voice Seed",
                        info="控制音色特征"
                    )
                    with gr.Row():
                        guest_speed = gr.Slider(
                            minimum=1, maximum=9, value=5, step=1,
                            label="⚡ Speed"
                        )
                        guest_emotion = gr.Slider(
                            minimum=0, maximum=9, value=5, step=1,
                            label="💫 Emotion"
                        )
                
                gr.HTML("<div style='height: 16px'></div>")
                
                # 全局设置
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    pause_duration = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                        label="Pause Duration (seconds)"
                    )
        
        # ===== 生成按钮 =====
        gr.HTML("<div style='height: 24px'></div>")
        
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                generate_btn = gr.Button(
                    "✨ Generate Podcast",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"]
                )
            with gr.Column(scale=1):
                pass
        
        # 生成状态
        generation_status = gr.HTML(
            value='<div style="text-align: center; color: #9ca3af; padding: 12px;">Ready to generate...</div>'
        )
        
        # ===== 音频播放器 =====
        gr.HTML("""
        <div class="panel-header" style="margin-top: 24px;">
            <div class="panel-icon">🎧</div>
            <span class="panel-title">Audio Output</span>
        </div>
        """)
        
        audio_output = gr.Audio(
            label="",
            type="filepath",
            interactive=False,
            show_label=False,
            elem_classes=["audio-player"]
        )
        
        # ===== 页脚 =====
        gr.HTML("""
        <div style="text-align: center; padding: 32px 0 16px; color: #9ca3af; font-size: 13px; border-top: 1px solid #e5e7eb; margin-top: 32px;">
            <strong>Deep Podcast</strong> · Powered by ChatTTS · 
            <a href="https://github.com/shaanguan/deep-podcast" style="color: #667eea; text-decoration: none;">GitHub</a>
        </div>
        """)
        
        # ===== 事件绑定 =====
        
        # 文件上传
        file_upload.change(
            fn=on_file_upload,
            inputs=file_upload,
            outputs=[text_input, stats_display]
        ).then(
            fn=lambda s: f'<div class="stats-display">{s}</div>' if s else '<div class="stats-display">📊 等待输入脚本...</div>',
            inputs=stats_display,
            outputs=stats_display
        )
        
        # 文本变化
        text_input.change(
            fn=on_text_change,
            inputs=text_input,
            outputs=stats_display
        ).then(
            fn=lambda s: f'<div class="stats-display">{s}</div>' if s else '<div class="stats-display">📊 等待输入脚本...</div>',
            inputs=stats_display,
            outputs=stats_display
        )
        
        # 加载示例
        load_sample_btn.click(
            fn=on_load_sample,
            outputs=[text_input, stats_display]
        ).then(
            fn=lambda s: f'<div class="stats-display">{s}</div>' if s else '<div class="stats-display">📊 等待输入脚本...</div>',
            inputs=stats_display,
            outputs=stats_display
        )
        
        # 生成
        def on_generate_wrapper(*args):
            audio_path, status = generate_audio(*args)
            status_html = f'<div style="text-align: center; padding: 12px; color: {"#166534" if "✓" in status else "#dc2626"}; font-weight: 500;">{status}</div>'
            return audio_path, status_html
        
        generate_btn.click(
            fn=on_generate_wrapper,
            inputs=[
                text_input,
                host_seed, host_speed, host_emotion,
                guest_seed, guest_speed, guest_emotion,
                pause_duration
            ],
            outputs=[audio_output, generation_status]
        )
    
    return app


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print()
    print("┌─────────────────────────────────────────┐")
    print("│     🎙️  Deep Podcast - Premium UI       │")
    print("│         AI Voice Studio v2.0            │")
    print("└─────────────────────────────────────────┘")
    print()
    
    check_hardware()
    print()
    
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_api=False
    )
