"""
工具函数模块
- 硬件检测（CUDA/MPS/CPU）
- 底噪生成与加载
- 音频处理（crossfade、归一化）
- 文本切分
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 终端输出工具
# ============================================================

def print_info(msg: str):
    """打印信息"""
    if HAS_COLORAMA:
        print(f"{Fore.CYAN}[*]{Style.RESET_ALL} {msg}")
    else:
        print(f"[*] {msg}")


def print_success(msg: str):
    """打印成功信息"""
    if HAS_COLORAMA:
        print(f"{Fore.GREEN}[✓]{Style.RESET_ALL} {msg}")
    else:
        print(f"[✓] {msg}")


def print_warning(msg: str):
    """打印警告信息"""
    if HAS_COLORAMA:
        print(f"{Fore.YELLOW}[!]{Style.RESET_ALL} {msg}")
    else:
        print(f"[!] {msg}")


def print_error(msg: str):
    """打印错误信息"""
    if HAS_COLORAMA:
        print(f"{Fore.RED}[✗]{Style.RESET_ALL} {msg}")
    else:
        print(f"[✗] {msg}")


# ============================================================
# 硬件检测
# ============================================================

def check_hardware() -> dict:
    """
    检测硬件环境，返回设备信息
    
    Returns:
        包含设备类型和详细信息的字典
    """
    info = {
        "device": "cpu",
        "device_name": "CPU",
        "cuda_available": False,
        "mps_available": False,
        "gpu_memory": None,
    }
    
    if not HAS_TORCH:
        print_warning("PyTorch 未安装，将使用 CPU")
        return info
    
    # 检查 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        
        # 获取显存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory"] = f"{total_memory / 1024**3:.1f} GB"
        
        print_success(f"检测到 NVIDIA GPU: {info['device_name']}")
        print_info(f"显存: {info['gpu_memory']}")
        return info
    
    # 检查 MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"
        
        print_success(f"检测到 {info['device_name']}")
        return info
    
    # 默认 CPU
    print_warning("未检测到 GPU，将使用 CPU（速度较慢）")
    return info


def get_device() -> str:
    """
    获取最佳可用设备
    
    Returns:
        设备字符串 ('cuda', 'mps', 或 'cpu')
    """
    if not HAS_TORCH:
        return "cpu"
    
    if torch.cuda.is_available():
        return "cuda"
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


# ============================================================
# 音频处理
# ============================================================

def generate_room_tone(
    duration: float,
    sample_rate: int = 24000,
    volume: float = 0.001  # 降低默认音量，0.01 太响
) -> np.ndarray:
    """
    生成拟真底噪（布朗噪音）
    
    白噪音太刺耳，使用累积求和模拟布朗噪音，听感更温暖
    真实录音环境底噪更接近布朗/粉红噪音（低沉的嗡嗡声）
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
        volume: 音量 (0.0-1.0)
        
    Returns:
        音频数据 numpy 数组
    """
    num_samples = int(duration * sample_rate)
    
    # 1. 生成白噪音
    white = np.random.randn(num_samples).astype(np.float32)
    
    # 2. 转换为布朗噪音（积分/累积求和）
    # 布朗噪音在高频衰减更快，听起来更柔和
    brown = np.cumsum(white)
    
    # 3. 去除直流分量（Center around 0）
    brown -= np.mean(brown)
    
    # 4. 归一化幅度
    max_val = np.max(np.abs(brown))
    if max_val > 0:
        brown /= max_val
    
    # 5. 应用音量
    return (brown * volume).astype(np.float32)


def load_room_tone(
    file_path: str,
    target_duration: float,
    sample_rate: int = 24000
) -> Optional[np.ndarray]:
    """
    加载底噪文件，支持循环拼接到目标时长
    
    Args:
        file_path: 底噪文件路径
        target_duration: 目标时长（秒）
        sample_rate: 目标采样率
        
    Returns:
        音频数据，加载失败返回 None
    """
    if not HAS_SOUNDFILE:
        return None
    
    if not os.path.exists(file_path):
        return None
    
    try:
        # 加载音频
        data, sr = sf.read(file_path)
        
        # 如果是多声道，转为单声道
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # 重采样（如果需要）
        if sr != sample_rate:
            # 简单的线性插值重采样
            original_length = len(data)
            target_length = int(original_length * sample_rate / sr)
            indices = np.linspace(0, original_length - 1, target_length)
            data = np.interp(indices, np.arange(original_length), data)
        
        # 计算目标样本数
        target_samples = int(target_duration * sample_rate)
        
        # 如果底噪不够长，循环拼接
        if len(data) < target_samples:
            repeats = int(np.ceil(target_samples / len(data)))
            data = np.tile(data, repeats)
        
        # 裁剪到目标长度
        data = data[:target_samples]
        
        return data.astype(np.float32)
        
    except Exception as e:
        print_warning(f"加载底噪文件失败: {e}")
        return None


def get_pause_audio(
    duration: float,
    sample_rate: int = 24000,
    use_room_tone: bool = True,
    room_tone_path: Optional[str] = None,
    room_tone_volume: float = 0.01
) -> np.ndarray:
    """
    获取停顿音频（底噪或静音）
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
        use_room_tone: 是否使用底噪
        room_tone_path: 底噪文件路径
        room_tone_volume: 底噪音量
        
    Returns:
        音频数据
    """
    if use_room_tone:
        # 尝试加载底噪文件
        if room_tone_path:
            room_tone = load_room_tone(room_tone_path, duration, sample_rate)
            if room_tone is not None:
                return room_tone * room_tone_volume
        
        # 生成白噪音底噪
        return generate_room_tone(duration, sample_rate, room_tone_volume)
    
    # 使用静音
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def crossfade_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    crossfade_samples: int
) -> np.ndarray:
    """
    对两段音频进行交叉淡入淡出拼接
    
    Args:
        audio1: 第一段音频
        audio2: 第二段音频
        crossfade_samples: 交叉淡化的样本数
        
    Returns:
        拼接后的音频
    """
    if crossfade_samples <= 0 or len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        # 无法进行 crossfade，直接拼接
        return np.concatenate([audio1, audio2])
    
    # 创建淡入淡出曲线
    fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
    
    # 应用淡出到 audio1 末尾
    audio1_fade = audio1.copy()
    audio1_fade[-crossfade_samples:] *= fade_out
    
    # 应用淡入到 audio2 开头
    audio2_fade = audio2.copy()
    audio2_fade[:crossfade_samples] *= fade_in
    
    # 混合重叠部分
    overlap = audio1_fade[-crossfade_samples:] + audio2_fade[:crossfade_samples]
    
    # 拼接结果
    result = np.concatenate([
        audio1_fade[:-crossfade_samples],
        overlap,
        audio2_fade[crossfade_samples:]
    ])
    
    return result


def normalize_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    target_lufs: float = -16.0
) -> np.ndarray:
    """
    使用 ITU-R BS.1770-4 标准进行响度归一化 (LUFS)
    
    专业播客标准：
    - Apple Podcasts / Spotify: -16 LUFS (立体声) 或 -19 LUFS (单声道)
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        target_lufs: 目标响度 (默认 -16 LUFS，播客行业标准)
        
    Returns:
        归一化后的音频
    """
    if len(audio) == 0:
        return audio
    
    # 优先使用 pyloudnorm（专业 LUFS 标准）
    if HAS_PYLOUDNORM:
        try:
            # 创建响度计量器
            meter = pyln.Meter(sample_rate)
            
            # 测量当前响度
            loudness = meter.integrated_loudness(audio)
            
            # 避免静音片段导致无限大增益
            if loudness == -float('inf') or np.isnan(loudness):
                return audio
            
            # 归一化到目标响度
            normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
            
            # 防止爆音 (Clipping protection)
            peak = np.max(np.abs(normalized))
            if peak > 1.0:
                normalized = normalized / peak * 0.99
            
            return normalized.astype(np.float32)
            
        except Exception:
            pass  # 回退到 RMS 方式
    
    # 回退方案：使用 RMS 归一化
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms < 1e-10:  # 避免除零
        return audio
    
    # 将 LUFS 近似转换为 RMS 目标值
    target_rms = 10 ** ((target_lufs + 10) / 20)  # 粗略近似
    
    # 计算并应用增益
    gain = target_rms / rms
    normalized = audio * gain
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized.astype(np.float32)


# ============================================================
# 文本处理
# ============================================================

def split_text(
    text: str,
    max_length: int = 100
) -> List[str]:
    """
    将长文本按标点符号切分为短句
    
    Args:
        text: 原始文本
        max_length: 最大长度（字符数）
        
    Returns:
        切分后的句子列表
    """
    if len(text) <= max_length:
        return [text]
    
    # 按标点符号切分
    # 保留分隔符
    pattern = r'([。！？!?;；,，])'
    parts = re.split(pattern, text)
    
    # 重新组合（将标点符号附加到前面的文本）
    sentences = []
    current = ""
    
    for i, part in enumerate(parts):
        if not part:
            continue
        
        # 如果是标点符号，附加到当前句子
        if re.match(pattern, part):
            current += part
        else:
            # 如果当前句子 + 新内容超过限制，先保存当前句子
            if current and len(current) + len(part) > max_length:
                sentences.append(current.strip())
                current = part
            else:
                current += part
    
    # 保存最后一个句子
    if current:
        sentences.append(current.strip())
    
    # 过滤空句子
    sentences = [s for s in sentences if s]
    
    # 如果某个句子仍然超长，强制切分
    final_sentences = []
    for s in sentences:
        if len(s) > max_length:
            # 强制按字数切分
            for i in range(0, len(s), max_length):
                chunk = s[i:i + max_length]
                if chunk:
                    final_sentences.append(chunk)
        else:
            final_sentences.append(s)
    
    return final_sentences


# ============================================================
# 缓存相关
# ============================================================

def get_cache_filename(
    text: str,
    speaker_id: str,
    seed: int,
    index: int
) -> str:
    """
    生成缓存文件名（基于内容的 MD5 哈希）
    
    Args:
        text: 文本内容
        speaker_id: 发言人 ID
        seed: 声纹种子
        index: 序号
        
    Returns:
        缓存文件名
    """
    # 创建唯一标识
    content = f"{text}|{speaker_id}|{seed}"
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    # 格式: 001_spk1_a1b2c3d4.wav
    return f"{index:03d}_spk{speaker_id}_{content_hash}.wav"


def clean_directory(directory: str, pattern: str = "*"):
    """
    清理目录中的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return
    
    import glob
    files = glob.glob(str(dir_path / pattern))
    
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print_warning(f"无法删除文件 {f}: {e}")


# ============================================================
# 混响处理
# ============================================================

def apply_reverb(
    audio: np.ndarray,
    sample_rate: int = 24000,
    room_size: float = 0.1,
    damping: float = 0.5,
    wet_level: float = 0.1
) -> np.ndarray:
    """
    应用简单的算法混响，统一空间感
    模拟 Schroeder 混响模型，让不同角色听起来像在同一个录音棚里
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        room_size: 房间大小 (0.0-1.0)，越大混响时间越长
        damping: 高频阻尼 (0.0-1.0)，越大声音越温暖
        wet_level: 湿声比例 (0.0-1.0)，混响强度
        
    Returns:
        添加混响后的音频
    """
    if not HAS_SCIPY or len(audio) == 0:
        return audio
    
    # 生成一个简单的房间脉冲响应 (Impulse Response)
    # 房间越大，混响时间(RT60)越长
    rt60 = 0.1 + (room_size * 0.5)  # 0.1s - 0.6s
    num_samples = int(rt60 * sample_rate)
    
    # 生成指数衰减的白噪音作为 IR
    t = np.linspace(0, 1, num_samples)
    ir = np.random.randn(num_samples) * np.exp(-t * (5 + damping * 5))
    
    # 归一化 IR
    ir = ir / np.max(np.abs(ir))
    
    # 使用 FFT 卷积 (比直接卷积快得多)
    wet_signal = scipy.signal.fftconvolve(audio, ir, mode='full')
    
    # 截断到原始长度 (为了对齐选原始长度)
    wet_signal = wet_signal[:len(audio)]
    
    # 混合干湿声
    # dry (1.0 - wet_level) + wet (wet_level)
    output = (audio * (1.0 - wet_level)) + (wet_signal * wet_level)
    
    # 防止削波
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
        
    return output.astype(np.float32)

