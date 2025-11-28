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

# 并发锁：防止多用户同时生成导致显存爆炸或状态混乱
generation_lock = threading.Lock()


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
    stop_requested: bool = False
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
        APP_STATE.progress = 0.02
        
        APP_STATE.config = load_config()
        APP_STATE.role_manager = RoleManager(APP_STATE.config)
        APP_STATE.generator = AudioGenerator(APP_STATE.config, APP_STATE.role_manager)
        
        APP_STATE.progress = 0.05
        
        temp_dir = str(PROJECT_ROOT / "temp")
        assets_dir = str(PROJECT_ROOT / "assets")
        
        if not APP_STATE.generator.initialize(temp_dir=temp_dir, assets_dir=assets_dir):
            return False, "模型加载失败"
        
        APP_STATE.model_loaded = True
        APP_STATE.progress = 0.1  # 模型加载完成占 10%
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


@app.route('/api/stop', methods=['POST'])
def stop_generation():
    """终止生成"""
    global APP_STATE
    APP_STATE.stop_requested = True
    APP_STATE.is_generating = False
    APP_STATE.status = "stopped"
    # 设置生成器的停止标志
    if APP_STATE.generator is not None:
        APP_STATE.generator.stop_flag = True
    return jsonify({"success": True})


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
    
    # 非阻塞获取锁，防止并发调用
    if not generation_lock.acquire(blocking=False):
        return jsonify({"success": False, "error": "当前有任务正在进行，请稍候..."})
    
    if APP_STATE.is_generating:
        generation_lock.release()
        return jsonify({"success": False, "error": "正在生成中"})
    
    data = request.json
    text = data.get('text', '')
    roles_config = data.get('roles', {})
    resume_mode = data.get('resume', False)  # 断点续传模式
    
    # 解析文本
    stats, dialogues = parse_text(text)
    if stats is None:
        return jsonify({"success": False, "error": dialogues})
    
    total_segs = len(dialogues)
    
    APP_STATE.is_generating = True
    APP_STATE.stop_requested = False
    APP_STATE.progress = 0
    APP_STATE.status = "generating"
    # 重置生成器的停止标志
    if APP_STATE.generator is not None:
        APP_STATE.generator.stop_flag = False
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
            used_seeds = set()
            
            # 获取原始配置中的 emb_file 映射（seed -> emb_file）
            original_roles = load_config().get('roles', {})
            seed_to_emb = {}
            for rid, rdata in original_roles.items():
                if 'emb_file' in rdata and rdata.get('seed'):
                    seed_to_emb[rdata['seed']] = rdata['emb_file']
            
            for role_id, role_data in roles_config.items():
                seed = role_data.get('seed', 3333)
                speed = role_data.get('speed', 5)
                oral = role_data.get('oral', 0)
                laugh = role_data.get('laugh', 0)
                break_ = role_data.get('break_', 3)
                
                # 确保 seed 严格不重复
                original_seed = seed
                while seed in used_seeds:
                    seed = seed + 1000
                    if seed > 9999:
                        seed = (seed % 9999) + 1
                used_seeds.add(seed)
                
                role_entry = {
                    "seed": seed,
                    "prompt": f"[speed_{speed}]",
                    "desc": f"角色{role_id}",
                    "refine_override": {
                        "oral": oral,
                        "laugh": laugh,
                        "break": break_
                    }
                }
                
                # 如果 seed 对应有 emb_file，添加到配置中
                if original_seed in seed_to_emb:
                    role_entry["emb_file"] = seed_to_emb[original_seed]
                    print(f"[DEBUG] 角色 {role_id} 使用 emb_file: {role_entry['emb_file']}")
                
                new_roles[role_id] = role_entry
            
            APP_STATE.config['roles'] = new_roles
            
            # 打印角色配置，用于调试
            print(f"[DEBUG] 角色配置: {new_roles}")
            
            APP_STATE.role_manager = RoleManager(APP_STATE.config)
            APP_STATE.generator.role_manager = APP_STATE.role_manager
            
            # 打印角色缓存，确认配置已加载
            print(f"[DEBUG] 角色缓存: {APP_STATE.role_manager.role_cache}")
            
            output_path = str(PROJECT_ROOT / "data" / "output" / "podcast.wav")
            
            # 清空 temp 目录（非断点续传模式）
            temp_dir = PROJECT_ROOT / "temp"
            if not resume_mode and temp_dir.exists():
                for f in temp_dir.glob("*.wav"):
                    try:
                        f.unlink()
                    except:
                        pass
            
            # 生成音频
            total = APP_STATE.total_segments
            
            # 记录开始时已有的文件数（用于断点续传模式）
            temp_dir_path = PROJECT_ROOT / "temp"
            initial_count = len(list(temp_dir_path.glob("*.wav"))) if temp_dir_path.exists() else 0
            APP_STATE.current_segment = initial_count
            
            # 监控进度：10%（模型已加载）-> 95%（生成完成）-> 100%（合并完成）
            stop_monitor = threading.Event()
            
            def monitor_progress():
                while not stop_monitor.is_set():
                    if temp_dir_path.exists():
                        current_count = len(list(temp_dir_path.glob("*.wav")))
                        APP_STATE.current_segment = current_count
                        if total > 0:
                            # 进度 = 10% + 85% * (完成数/总数)
                            progress = 0.1 + (current_count / total) * 0.85
                            APP_STATE.progress = min(progress, 0.95)
                    time.sleep(0.3)
            
            monitor_thread = threading.Thread(target=monitor_progress)
            monitor_thread.start()
            
            success = APP_STATE.generator.generate(
                dialogues, output_path, resume=resume_mode, force=not resume_mode
            )
            
            # 停止监控线程
            stop_monitor.set()
            monitor_thread.join(timeout=2)
            
            # 先设置状态，再设置 is_generating = False，避免竞态条件
            if success and os.path.exists(output_path):
                APP_STATE.progress = 1.0
                APP_STATE.current_segment = total
                APP_STATE.status = "complete"
            else:
                APP_STATE.status = "error"
                APP_STATE.message = "生成失败"
            
            APP_STATE.is_generating = False
            generation_lock.release()  # 释放锁
                
        except Exception as e:
            APP_STATE.status = "error"
            APP_STATE.message = str(e)
            APP_STATE.is_generating = False
            generation_lock.release()  # 异常时也要释放锁
    
    thread = threading.Thread(target=run_generation)
    thread.start()
    
    return jsonify({"success": True, "message": "生成任务已启动", "total_segments": total_segs})


@app.route('/api/audio')
def get_audio():
    audio_path = PROJECT_ROOT / "data" / "output" / "podcast.wav"
    if audio_path.exists():
        return send_file(str(audio_path), mimetype='audio/wav')
    return jsonify({"error": "音频文件不存在"}), 404


@app.route('/api/audio/waveform/<audio_id>')
def get_audio_waveform(audio_id):
    """获取音频波形数据（最多15秒）"""
    import numpy as np
    import soundfile as sf
    
    history_path = PROJECT_ROOT / "data" / "history" / f"{audio_id}.wav"
    if not history_path.exists():
        return jsonify({"success": False, "error": "文件不存在"})
    
    try:
        # 读取音频
        audio, sr = sf.read(str(history_path))
        
        # 最多取15秒
        max_samples = sr * 15
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # 如果是立体声，转为单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # 将音频分成 80 个区间，计算每个区间的平均振幅
        num_bars = 80
        chunk_size = len(audio) // num_bars
        
        waveform = []
        for i in range(num_bars):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio[start:end]
            # 计算 RMS 振幅
            rms = np.sqrt(np.mean(chunk ** 2))
            waveform.append(float(rms))
        
        # 归一化到 0-1
        max_val = max(waveform) if waveform else 1
        if max_val > 0:
            waveform = [v / max_val for v in waveform]
        
        return jsonify({"success": True, "waveform": waveform})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/audio/save', methods=['POST'])
def save_audio_to_history():
    """保存当前音频到历史记录"""
    try:
        audio_path = PROJECT_ROOT / "data" / "output" / "podcast.wav"
        if not audio_path.exists():
            return jsonify({"success": False, "error": "音频文件不存在"})
        
        # 创建历史目录
        history_dir = PROJECT_ROOT / "data" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一 ID
        audio_id = str(int(time.time() * 1000))
        
        # 复制文件
        import shutil
        dest_path = history_dir / f"{audio_id}.wav"
        shutil.copy2(str(audio_path), str(dest_path))
        
        return jsonify({"success": True, "id": audio_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/audio/history/<audio_id>')
def get_history_audio(audio_id):
    """获取历史音频"""
    history_path = PROJECT_ROOT / "data" / "history" / f"{audio_id}.wav"
    if history_path.exists():
        return send_file(str(history_path), mimetype='audio/wav')
    return jsonify({"error": "历史音频不存在"}), 404


@app.route('/api/audio/history/<audio_id>', methods=['DELETE'])
def delete_history_audio(audio_id):
    """删除历史音频"""
    try:
        history_path = PROJECT_ROOT / "data" / "history" / f"{audio_id}.wav"
        if history_path.exists():
            os.remove(str(history_path))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/preview', methods=['POST'])
def preview_voice():
    """生成音色试听（带缓存）"""
    global APP_STATE
    
    try:
        data = request.json
        seed = int(data.get('seed', 3333))
        speed = int(data.get('speed', 5))
        oral = int(data.get('oral', 0))
        laugh = int(data.get('laugh', 0))
        break_ = int(data.get('break_', 3))
        
        # 生成缓存文件名（基于所有参数）
        cache_name = f"preview_{seed}_{speed}_{oral}_{laugh}_{break_}.wav"
        preview_dir = PROJECT_ROOT / "temp" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        cache_path = preview_dir / cache_name
        
        # 如果缓存存在，直接返回
        if cache_path.exists():
            return send_file(str(cache_path), mimetype='audio/wav')
        
        # 加载模型
        success, msg = ensure_model_loaded()
        if not success:
            return jsonify({"error": msg}), 500
        
        # 构建临时角色配置
        from src.role_manager import RoleConfig
        
        role = RoleConfig(
            seed=seed,
            prompt=f"[speed_{speed}]",
            desc="试听",
            oral=oral,
            laugh=laugh,
            break_level=break_,
            is_auto_generated=False
        )
        
        # 生成音频
        text = "大家好，欢迎收听本期播客"
        spk_emb = APP_STATE.generator._get_speaker_embedding(seed)
        audio = APP_STATE.generator._generate_single(text, role, spk_emb)
        
        if audio is None or len(audio) == 0:
            return jsonify({"error": "生成失败"}), 500
        
        # 保存到缓存文件
        sf.write(str(cache_path), audio, APP_STATE.generator.sample_rate)
        
        return send_file(str(cache_path), mimetype='audio/wav')
        
    except Exception as e:
        print(f"[试听错误] {e}")
        return jsonify({"error": str(e)}), 500


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
    
    app.run(host='127.0.0.1', port=5001, debug=True, threaded=True)
