#!/usr/bin/env python3
"""
Deep Podcast - Professional Web UI
基于 Gradio 的专业级可视化界面
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
    current_progress: float = 0.0
    current_status: str = ""
    
APP_STATE = AppState()


# ============================================================
# 配置管理
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


# ============================================================
# 模型管理
# ============================================================

def ensure_model_loaded(progress_callback=None) -> Tuple[bool, str]:
    """确保模型已加载（智能加载）"""
    global APP_STATE
    
    if APP_STATE.model_loaded and APP_STATE.generator is not None:
        return True, "模型就绪"
    
    try:
        if progress_callback:
            progress_callback(0.1, "正在加载配置...")
        
        APP_STATE.config = load_config()
        APP_STATE.role_manager = RoleManager(APP_STATE.config)
        APP_STATE.generator = AudioGenerator(APP_STATE.config, APP_STATE.role_manager)
        
        if progress_callback:
            progress_callback(0.2, "正在加载 ChatTTS 模型（首次约需 30 秒）...")
        
        temp_dir = str(PROJECT_ROOT / "temp")
        assets_dir = str(PROJECT_ROOT / "assets")
        
        if not APP_STATE.generator.initialize(temp_dir=temp_dir, assets_dir=assets_dir):
            return False, "模型加载失败"
        
        APP_STATE.model_loaded = True
        
        if progress_callback:
            progress_callback(0.3, "模型加载完成！")
        
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
                return "❌ 请安装 python-docx: pip install python-docx"
            doc = Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return '\n\n'.join(paragraphs)
        
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lines = []
            if isinstance(data, dict) and 'dialogues' in data:
                for item in data['dialogues']:
                    speaker = item.get('speaker', item.get('speaker_id', '1'))
                    text = item.get('text', item.get('content', ''))
                    lines.append(f"发言人{speaker} {text}")
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        speaker = item.get('speaker', item.get('speaker_id', '1'))
                        text = item.get('text', item.get('content', ''))
                        lines.append(f"发言人{speaker} {text}")
            elif isinstance(data, dict):
                for speaker, texts in data.items():
                    if isinstance(texts, list):
                        for text in texts:
                            lines.append(f"发言人{speaker} {text}")
                    else:
                        lines.append(f"发言人{speaker} {texts}")
            return '\n\n'.join(lines)
        
        elif file_ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = []
            current_speaker = None
            current_text = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith(('---', '***', '===', '```')):
                    continue
                
                # 检测发言人模式
                speaker_match = re.match(
                    r'^(?:#{1,6}\s*|\*\*)?(发言人|Speaker|说话人)\s*(\d+)\**\s*[:：]?\s*(.*)$',
                    line, re.IGNORECASE
                )
                if speaker_match:
                    if current_speaker and current_text:
                        lines.append(f"发言人{current_speaker} {''.join(current_text)}")
                    current_speaker = speaker_match.group(2)
                    current_text = [speaker_match.group(3)] if speaker_match.group(3) else []
                    continue
                
                # 清理 Markdown 格式
                line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
                line = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', line)
                line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
                line = re.sub(r'\*([^*]+)\*', r'\1', line)
                line = re.sub(r'`([^`]+)`', r'\1', line)
                
                if current_speaker and line:
                    current_text.append(line)
            
            if current_speaker and current_text:
                lines.append(f"发言人{current_speaker} {''.join(current_text)}")
            
            return '\n\n'.join(lines) if lines else content
        
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        return f"❌ 文件解析失败: {str(e)}"


def parse_text_to_dialogues(text: str) -> Tuple[str, list]:
    """解析文本为对话列表"""
    if not text.strip():
        return "⚠️ 请输入文本内容", []
    
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
            return "⚠️ 未解析到有效对话，请检查格式", []
        
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
        
        # 格式化输出
        preview = f"""╔══════════════════════════════════════════════════════════╗
║                     📊 解析统计                          ║
╠══════════════════════════════════════════════════════════╣
║  👥 角色数量    │  {len(speaker_stats):>3} 位                               ║
║  💬 对话片段    │  {len(dialogues):>3} 段                               ║
║  📝 总字数      │  {total_chars:>5} 字                             ║
║  ⏱️  预估时长    │  {est_minutes:>2}分{est_seconds:02d}秒                            ║
╠══════════════════════════════════════════════════════════╣
║                     📋 角色明细                          ║
╠══════════════════════════════════════════════════════════╣"""
        
        for spk_id in sorted(speaker_stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            stats = speaker_stats[spk_id]
            preview += f"\n║  发言人{spk_id:<3}     │  {stats['count']:>3} 段 / {stats['chars']:>5} 字                    ║"
        
        preview += f"""
╠══════════════════════════════════════════════════════════╣
║                     📜 对话预览                          ║
╠══════════════════════════════════════════════════════════╣"""
        
        for d in dialogues[:8]:
            text_preview = d.text[:30] + "..." if len(d.text) > 30 else d.text
            preview += f"\n║  [{d.index + 1:02d}] 发言人{d.speaker_id}: {text_preview:<35}║"
        
        if len(dialogues) > 8:
            preview += f"\n║  ... 还有 {len(dialogues) - 8} 条对话                                    ║"
        
        preview += "\n╚══════════════════════════════════════════════════════════╝"
        
        return preview, dialogues
        
    finally:
        os.unlink(temp_path)


# ============================================================
# 核心生成逻辑（带进度）
# ============================================================

def generate_with_progress(
    text: str,
    host_seed: int,
    host_speed: int,
    guest_seed: int,
    guest_speed: int,
    auto_seed_base: int,
    pause_duration: float,
    normalize: bool,
    temperature: float,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], str]:
    """带进度的音频生成"""
    global APP_STATE
    
    if APP_STATE.is_generating:
        return None, "⚠️ 已有任务在生成中，请稍候..."
    
    APP_STATE.is_generating = True
    start_time = time.time()
    
    try:
        # Step 1: 确保模型加载
        progress(0.05, desc="🔄 检查模型状态...")
        
        success, msg = ensure_model_loaded(
            lambda p, s: progress(p * 0.25, desc=s)
        )
        
        if not success:
            return None, f"❌ {msg}"
        
        # Step 2: 更新配置
        progress(0.25, desc="⚙️ 应用配置参数...")
        
        APP_STATE.config['roles'] = {
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
        APP_STATE.config['auto_seed']['base'] = auto_seed_base
        APP_STATE.config['audio']['pause_duration'] = pause_duration
        APP_STATE.config['audio']['normalize'] = normalize
        APP_STATE.config['style']['temperature'] = temperature
        
        APP_STATE.role_manager = RoleManager(APP_STATE.config)
        APP_STATE.generator.role_manager = APP_STATE.role_manager
        APP_STATE.generator.pause_duration = pause_duration
        APP_STATE.generator.do_normalize = normalize
        APP_STATE.generator.temperature = temperature
        
        # Step 3: 解析文本
        progress(0.28, desc="📝 解析对话文本...")
        _, dialogues = parse_text_to_dialogues(text)
        
        if not dialogues:
            return None, "❌ 解析失败，请检查文本格式"
        
        total_dialogues = len(dialogues)
        total_chars = sum(len(d.text) for d in dialogues)
        
        # 估算每个片段的时间（基于实际测试约 2-5 秒/片段）
        estimated_per_segment = 3.0
        estimated_total = total_dialogues * estimated_per_segment
        
        # Step 4: 逐段生成（模拟进度）
        progress(0.30, desc=f"🎙️ 开始生成 {total_dialogues} 个音频片段...")
        
        output_path = str(PROJECT_ROOT / "data" / "output" / "podcast_ui.wav")
        
        # 启动生成
        # 由于 generator.generate 是同步的，我们用线程来更新进度
        generation_complete = threading.Event()
        generation_result = {"success": False, "error": None}
        
        def run_generation():
            try:
                result = APP_STATE.generator.generate(
                    dialogues,
                    output_path,
                    resume=False,
                    force=True
                )
                generation_result["success"] = result
            except Exception as e:
                generation_result["error"] = str(e)
            finally:
                generation_complete.set()
        
        # 启动生成线程
        gen_thread = threading.Thread(target=run_generation)
        gen_thread.start()
        
        # 模拟进度更新
        progress_start = 0.30
        progress_end = 0.90
        progress_range = progress_end - progress_start
        
        segment_idx = 0
        while not generation_complete.is_set():
            # 检查 temp 目录中的文件数量来估算进度
            temp_dir = PROJECT_ROOT / "temp"
            if temp_dir.exists():
                wav_files = list(temp_dir.glob("*.wav"))
                segment_idx = len(wav_files)
            
            current_progress = progress_start + (segment_idx / max(total_dialogues, 1)) * progress_range
            current_progress = min(current_progress, progress_end)
            
            elapsed = time.time() - start_time
            if segment_idx > 0:
                estimated_remaining = (elapsed / segment_idx) * (total_dialogues - segment_idx)
                remaining_str = f"{int(estimated_remaining)}秒"
            else:
                remaining_str = "计算中..."
            
            progress(
                current_progress,
                desc=f"🎙️ 生成中 [{segment_idx}/{total_dialogues}] - 预计剩余: {remaining_str}"
            )
            
            time.sleep(0.5)
        
        gen_thread.join()
        
        # Step 5: 后处理
        if generation_result["error"]:
            return None, f"❌ 生成出错: {generation_result['error']}"
        
        if not generation_result["success"]:
            return None, "❌ 生成失败，请检查日志"
        
        progress(0.95, desc="🔊 音频后处理...")
        
        # 检查输出文件
        if not os.path.exists(output_path):
            return None, "❌ 输出文件未生成"
        
        # 计算时长
        data, sr = sf.read(output_path)
        duration = len(data) / sr
        minutes = int(duration // 60)
        seconds = duration % 60
        
        elapsed_total = time.time() - start_time
        elapsed_min = int(elapsed_total // 60)
        elapsed_sec = int(elapsed_total % 60)
        
        progress(1.0, desc="✅ 生成完成！")
        
        result_msg = f"""╔══════════════════════════════════════════════════════════╗
║                     ✅ 生成成功                          ║
╠══════════════════════════════════════════════════════════╣
║  🎵 音频时长    │  {minutes:>2}分{seconds:04.1f}秒                          ║
║  ⏱️  耗时        │  {elapsed_min:>2}分{elapsed_sec:02d}秒                            ║
║  📊 片段数量    │  {total_dialogues:>3} 段                               ║
║  📝 总字数      │  {total_chars:>5} 字                             ║
╚══════════════════════════════════════════════════════════╝"""
        
        return output_path, result_msg
        
    except Exception as e:
        return None, f"❌ 生成出错: {str(e)}"
    
    finally:
        APP_STATE.is_generating = False


# ============================================================
# UI 回调
# ============================================================

def on_parse_text(text: str):
    """解析文本"""
    preview, _ = parse_text_to_dialogues(text)
    return preview


def on_file_upload(file):
    """文件上传"""
    if file is None:
        return "", "⚠️ 请选择文件"
    
    file_path = file.name if hasattr(file, 'name') else file
    text = parse_uploaded_file(file_path)
    
    if text.startswith("❌"):
        return "", text
    
    preview, _ = parse_text_to_dialogues(text)
    return text, preview


def on_load_sample():
    """加载示例"""
    sample_path = PROJECT_ROOT / "data" / "input" / "transcript.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            text = f.read()
        preview, _ = parse_text_to_dialogues(text)
        return text, preview
    return "", "⚠️ 示例文件不存在"


def get_model_status():
    """获取模型状态"""
    if APP_STATE.model_loaded:
        return "🟢 模型已就绪"
    return "🟡 模型未加载（首次生成时自动加载）"


# ============================================================
# UI 界面
# ============================================================

def create_ui():
    """创建专业级 UI"""
    
    # 深色主题 CSS
    custom_css = """
    /* 全局样式 */
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    
    /* 标题区 */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        border: 1px solid #333;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d9ff 0%, #00ff88 50%, #ffcc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0.5rem;
        letter-spacing: 2px;
    }
    
    .subtitle {
        color: #888;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .version-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 0.75rem;
        font-weight: 600;
    }
    
    /* 卡片样式 */
    .card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00d9ff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* 状态指示器 */
    .status-indicator {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        text-align: center;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #0d4d1c 0%, #1a5928 100%);
        border: 1px solid #2d7a3e;
        color: #7dff9e;
    }
    
    .status-pending {
        background: linear-gradient(135deg, #4d3d0d 0%, #5a4a1a 100%);
        border: 1px solid #7a6a2d;
        color: #ffd97d;
    }
    
    /* 按钮样式 */
    .generate-btn {
        background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.3);
    }
    
    /* 进度条 */
    .progress-container {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* 音频播放器 */
    .audio-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
    }
    
    /* 文本框 */
    textarea {
        font-family: 'SF Mono', 'Menlo', monospace !important;
        font-size: 0.9rem !important;
    }
    
    /* 统计信息框 */
    .stats-box {
        font-family: 'SF Mono', 'Menlo', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        background: #0d0d14;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
    }
    """
    
    with gr.Blocks(
        title="Deep Podcast - AI 配音生成器",
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0d0d14",
            body_background_fill_dark="#0d0d14",
            block_background_fill="#1a1a2e",
            block_background_fill_dark="#1a1a2e",
            block_border_color="#333",
            block_border_color_dark="#333",
            input_background_fill="#16213e",
            input_background_fill_dark="#16213e",
            button_primary_background_fill="linear-gradient(135deg, #00d9ff 0%, #00ff88 100%)",
            button_primary_text_color="#000",
        )
    ) as app:
        
        # 标题区
        gr.HTML("""
        <div class="title-container">
            <h1 class="main-title">🎙️ DEEP PODCAST</h1>
            <p class="subtitle">AI-Powered Multi-Voice Podcast Generator</p>
            <span class="version-badge">v2.0 PRO</span>
        </div>
        """)
        
        # 主内容区
        with gr.Row(equal_height=False):
            
            # 左栏：输入
            with gr.Column(scale=5):
                
                gr.Markdown("### 📝 对话脚本")
                
                # 文件上传
                with gr.Accordion("📁 导入文件 (支持 Word/JSON/Markdown/TXT)", open=False):
                    file_upload = gr.File(
                        label="拖拽或点击上传",
                        file_types=[".docx", ".json", ".md", ".txt"],
                        file_count="single"
                    )
                
                text_input = gr.Textbox(
                    label="",
                    placeholder="""在此输入对话脚本，格式示例：

发言人1 大家好，欢迎收听本期播客！今天我们来聊聊人工智能。

发言人2 没错，AI 技术发展得太快了，你觉得未来会怎样？

发言人1 我认为 AI 会成为我们生活中不可或缺的一部分...""",
                    lines=12,
                    max_lines=20
                )
                
                with gr.Row():
                    load_sample_btn = gr.Button("📄 加载示例", size="sm", variant="secondary")
                    parse_btn = gr.Button("🔍 解析预览", size="sm", variant="secondary")
                
                parse_output = gr.Textbox(
                    label="解析结果",
                    lines=12,
                    interactive=False,
                    elem_classes=["stats-box"]
                )
            
            # 右栏：配置
            with gr.Column(scale=4):
                
                gr.Markdown("### ⚙️ 生成配置")
                
                # 角色配置
                with gr.Group():
                    gr.Markdown("**🎭 角色声音**")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("<small>**主持人 (发言人1)**</small>")
                            host_seed = gr.Slider(
                                minimum=1, maximum=10000, value=3333, step=1,
                                label="声纹种子"
                            )
                            host_speed = gr.Slider(
                                minimum=1, maximum=9, value=5, step=1,
                                label="语速 (1慢-9快)"
                            )
                        
                        with gr.Column():
                            gr.Markdown("<small>**嘉宾 (发言人2)**</small>")
                            guest_seed = gr.Slider(
                                minimum=1, maximum=10000, value=5674, step=1,
                                label="声纹种子"
                            )
                            guest_speed = gr.Slider(
                                minimum=1, maximum=9, value=5, step=1,
                                label="语速 (1慢-9快)"
                            )
                    
                    auto_seed_base = gr.Slider(
                        minimum=1000, maximum=9000, value=5000, step=100,
                        label="其他角色种子基数",
                        info="发言人N的种子 = 基数 + N×337"
                    )
                
                # 音频配置
                with gr.Group():
                    gr.Markdown("**🔊 音频参数**")
                    
                    with gr.Row():
                        pause_duration = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                            label="段落停顿 (秒)"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                            label="语调变化度"
                        )
                    
                    normalize = gr.Checkbox(
                        value=True,
                        label="🔈 启用音频响度归一化 (LUFS 广播标准)"
                    )
                
                # 模型状态
                model_status = gr.Textbox(
                    label="模型状态",
                    value=get_model_status(),
                    interactive=False,
                    lines=1
                )
        
        # 生成按钮（居中大按钮）
        gr.Markdown("---")
        
        with gr.Row():
            gr.Column(scale=1)
            generate_btn = gr.Button(
                "🚀 一键生成播客",
                variant="primary",
                size="lg",
                scale=2,
                elem_classes=["generate-btn"]
            )
            gr.Column(scale=1)
        
        # 生成状态
        generation_status = gr.Textbox(
            label="生成状态",
            value="等待开始...",
            interactive=False,
            lines=8,
            elem_classes=["stats-box"]
        )
        
        # 音频输出
        gr.Markdown("### 🎧 生成结果")
        
        audio_output = gr.Audio(
            label="播放生成的播客",
            type="filepath",
            interactive=False,
            elem_classes=["audio-container"]
        )
        
        # 绑定事件
        file_upload.change(
            fn=on_file_upload,
            inputs=file_upload,
            outputs=[text_input, parse_output]
        )
        
        load_sample_btn.click(
            fn=on_load_sample,
            outputs=[text_input, parse_output]
        )
        
        parse_btn.click(
            fn=on_parse_text,
            inputs=text_input,
            outputs=parse_output
        )
        
        generate_btn.click(
            fn=generate_with_progress,
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
        gr.HTML("""
        <div style="text-align: center; color: #555; font-size: 0.85rem; padding: 2rem 0; border-top: 1px solid #333; margin-top: 2rem;">
            <strong>Deep Podcast v2.0</strong> | Powered by ChatTTS | 
            <a href="https://github.com/shaanguan/deep-podcast" target="_blank" style="color: #00d9ff;">GitHub</a>
        </div>
        """)
    
    return app


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           🎙️  Deep Podcast - Professional UI             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # 检测硬件
    hw_info = check_hardware()
    print()
    
    # 创建并启动 UI
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_api=False
    )
