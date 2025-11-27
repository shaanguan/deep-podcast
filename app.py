#!/usr/bin/env python3
"""
Deep Podcast - Professional Web Application
Flask + Custom Frontend
"""

import os
import sys
import json
import time
import tempfile
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import yaml

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

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import parse_transcript
from src.role_manager import RoleManager
from src.generator import AudioGenerator
from src.utils import check_hardware

app = Flask(__name__)
CORS(app)


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
    progress: float = 0.0
    status: str = "idle"
    total_segments: int = 0
    current_segment: int = 0
    message: str = ""
    
APP_STATE = AppState()


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_model_loaded() -> tuple:
    global APP_STATE
    
    if APP_STATE.model_loaded and APP_STATE.generator is not None:
        return True, "模型已就绪"
    
    try:
        APP_STATE.status = "loading"
        APP_STATE.progress = 0.1
        
        APP_STATE.config = load_config()
        APP_STATE.role_manager = RoleManager(APP_STATE.config)
        APP_STATE.generator = AudioGenerator(APP_STATE.config, APP_STATE.role_manager)
        
        APP_STATE.progress = 0.3
        
        temp_dir = str(PROJECT_ROOT / "temp")
        assets_dir = str(PROJECT_ROOT / "assets")
        
        if not APP_STATE.generator.initialize(temp_dir=temp_dir, assets_dir=assets_dir):
            return False, "模型加载失败"
        
        APP_STATE.model_loaded = True
        APP_STATE.progress = 1.0
        APP_STATE.status = "ready"
        return True, "模型加载成功"
        
    except Exception as e:
        APP_STATE.status = "error"
        return False, str(e)


def parse_text(text: str) -> tuple:
    if not text.strip():
        return None, "请输入脚本内容"
    
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
            return None, "未解析到有效对话"
        
        # 统计
        speakers = {}
        total_chars = 0
        
        for d in dialogues:
            if d.speaker_id not in speakers:
                speakers[d.speaker_id] = {"count": 0, "chars": 0}
            speakers[d.speaker_id]["count"] += 1
            speakers[d.speaker_id]["chars"] += len(d.text)
            total_chars += len(d.text)
        
        estimated_seconds = total_chars * 0.3
        
        return {
            "dialogues": len(dialogues),
            "speakers": len(speakers),
            "chars": total_chars,
            "duration": int(estimated_seconds),
            "speaker_details": speakers
        }, dialogues
        
    finally:
        os.unlink(temp_path)


# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    return jsonify({
        "model_loaded": APP_STATE.model_loaded,
        "is_generating": APP_STATE.is_generating,
        "progress": APP_STATE.progress,
        "status": APP_STATE.status,
        "total_segments": APP_STATE.total_segments,
        "current_segment": APP_STATE.current_segment,
        "message": APP_STATE.message
    })


@app.route('/api/parse', methods=['POST'])
def api_parse():
    data = request.json
    text = data.get('text', '')
    
    stats, dialogues = parse_text(text)
    
    if stats is None:
        return jsonify({"success": False, "error": dialogues})
    
    return jsonify({"success": True, "stats": stats})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    global APP_STATE
    
    if APP_STATE.is_generating:
        return jsonify({"success": False, "error": "正在生成中"})
    
    data = request.json
    text = data.get('text', '')
    roles_config = data.get('roles', {})
    
    # 解析文本
    stats, dialogues = parse_text(text)
    if stats is None:
        return jsonify({"success": False, "error": dialogues})
    
    total_segs = len(dialogues)
    
    APP_STATE.is_generating = True
    APP_STATE.progress = 0
    APP_STATE.status = "generating"
    APP_STATE.total_segments = total_segs
    APP_STATE.current_segment = 0
    APP_STATE.message = ""
    
    def run_generation():
        global APP_STATE
        try:
            # 加载模型
            APP_STATE.status = "loading"
            success, msg = ensure_model_loaded()
            if not success:
                APP_STATE.status = "error"
                APP_STATE.message = msg
                APP_STATE.is_generating = False
                return
            
            APP_STATE.status = "generating"
            
            # 根据前端传来的角色配置构建
            new_roles = {}
            for role_id, role_data in roles_config.items():
                seed = role_data.get('seed', 3333)
                speed = role_data.get('speed', 5)
                emotion = role_data.get('emotion', 4)
                gender = role_data.get('gender', 'male')
                
                new_roles[role_id] = {
                    "seed": seed,
                    "prompt": f"[speed_{speed}]",
                    "desc": f"角色{role_id}",
                    "gender": gender,
                    "refine_override": {
                        "oral": min(emotion, 5),
                        "laugh": max(0, emotion - 5),
                        "break": 3 if gender == 'male' else 4
                    }
                }
            
            APP_STATE.config['roles'] = new_roles
            
            APP_STATE.role_manager = RoleManager(APP_STATE.config)
            APP_STATE.generator.role_manager = APP_STATE.role_manager
            
            output_path = str(PROJECT_ROOT / "data" / "output" / "podcast.wav")
            
            # 生成
            APP_STATE.progress = 0.1
            
            total = APP_STATE.total_segments
            
            # 监控进度
            def monitor_progress():
                while APP_STATE.is_generating:
                    temp_dir = PROJECT_ROOT / "temp"
                    if temp_dir.exists():
                        done = len(list(temp_dir.glob("*.wav")))
                        APP_STATE.current_segment = done
                        if total > 0:
                            APP_STATE.progress = 0.1 + (done / total) * 0.85
                    time.sleep(0.5)
            
            monitor_thread = threading.Thread(target=monitor_progress)
            monitor_thread.start()
            
            success = APP_STATE.generator.generate(
                dialogues, output_path, resume=False, force=True
            )
            
            APP_STATE.is_generating = False
            monitor_thread.join()
            
            if success and os.path.exists(output_path):
                APP_STATE.progress = 1.0
                APP_STATE.current_segment = total
                APP_STATE.status = "complete"
            else:
                APP_STATE.status = "error"
                APP_STATE.message = "生成失败"
                
        except Exception as e:
            APP_STATE.status = "error"
            APP_STATE.message = str(e)
            APP_STATE.is_generating = False
    
    thread = threading.Thread(target=run_generation)
    thread.start()
    
    return jsonify({"success": True, "message": "生成任务已启动", "total_segments": total_segs})


@app.route('/api/audio')
def get_audio():
    audio_path = PROJECT_ROOT / "data" / "output" / "podcast.wav"
    if audio_path.exists():
        return send_file(str(audio_path), mimetype='audio/wav')
    return jsonify({"error": "音频文件不存在"}), 404


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "没有文件"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "没有选择文件"})
    
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.docx'):
            if not HAS_DOCX:
                return jsonify({"success": False, "error": "不支持 Word 文件"})
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                file.save(tmp.name)
                doc = Document(tmp.name)
                text = '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
                os.unlink(tmp.name)
        
        elif filename.endswith('.json'):
            data = json.load(file)
            lines = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        speaker = item.get('speaker', '1')
                        content = item.get('text', '')
                        lines.append(f"发言人{speaker} {content}")
            text = '\n\n'.join(lines)
        
        else:
            text = file.read().decode('utf-8')
        
        return jsonify({"success": True, "text": text})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print()
    print("╔════════════════════════════════════════════╗")
    print("║      🎙️  Deep Podcast · Professional      ║")
    print("╚════════════════════════════════════════════╝")
    print()
    
    check_hardware()
    print()
    print("启动服务器: http://localhost:5001")
    print()
    
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
