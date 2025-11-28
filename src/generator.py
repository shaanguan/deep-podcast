"""
音频生成器模块
- ChatTTS 调用封装
- 断点续传支持
- 分段保存与合并
"""

import os
import random
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ChatTTS
    HAS_CHATTTS = True
except ImportError:
    HAS_CHATTTS = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .parser import DialogueItem
from .role_manager import RoleManager, RoleConfig
from .utils import (
    print_info, print_success, print_warning, print_error,
    get_device, check_hardware,
    split_text, get_cache_filename,
    get_pause_audio, crossfade_audio, normalize_audio,
    apply_reverb
)


class AudioGenerator:
    """音频生成器"""
    
    def __init__(self, config: Dict[str, Any], role_manager: RoleManager):
        """
        初始化音频生成器
        
        Args:
            config: 完整配置字典
            role_manager: 角色管理器实例
        """
        self.config = config
        self.role_manager = role_manager
        self.stop_flag = False  # 停止标志
        
        # 音频配置
        audio_config = config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 24000)
        self.pause_duration = audio_config.get('pause_duration', 0.5)
        self.do_normalize = audio_config.get('normalize', True)
        self.use_room_tone = audio_config.get('use_room_tone', True)
        self.room_tone_volume = audio_config.get('room_tone_volume', 0.01)
        self.crossfade_ms = audio_config.get('crossfade_ms', 50)
        
        # 文本配置
        text_config = config.get('text', {})
        self.max_chunk_length = text_config.get('max_chunk_length', 100)
        
        # 样式配置
        style_config = config.get('style', {})
        self.temperature = style_config.get('temperature', 0.3)
        self.top_P = style_config.get('top_P', 0.7)
        self.top_K = style_config.get('top_K', 20)
        
        # ChatTTS 实例
        self.chat: Optional[Any] = None
        self.device: str = "cpu"
        
        # 路径
        self.temp_dir: Optional[Path] = None
        self.assets_dir: Optional[Path] = None
    
    def initialize(
        self,
        temp_dir: str = "temp",
        assets_dir: str = "assets"
    ) -> bool:
        """
        初始化 ChatTTS 模型
        
        Args:
            temp_dir: 缓存目录
            assets_dir: 资源目录
            
        Returns:
            是否初始化成功
        """
        if not HAS_CHATTTS:
            print_error("ChatTTS 未安装，请运行: pip install ChatTTS")
            return False
        
        if not HAS_TORCH:
            print_error("PyTorch 未安装，请运行: pip install torch")
            return False
        
        # 检测硬件
        print_info("检测硬件环境...")
        hw_info = check_hardware()
        self.device = hw_info['device']
        
        # 设置目录
        self.temp_dir = Path(temp_dir)
        self.assets_dir = Path(assets_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChatTTS
        print_info("正在加载 ChatTTS 模型...")
        try:
            self.chat = ChatTTS.Chat()
            self.chat.load(compile=False)
            print_success("ChatTTS 模型加载完成")
            return True
        except Exception as e:
            print_error(f"ChatTTS 加载失败: {e}")
            return False
    
    def _set_all_seeds(self, seed: int):
        """
        设置所有随机种子，确保音色稳定
        
        Args:
            seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # 设置 deterministic 模式
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_speaker_embedding(self, seed: int, emb_file: str = None) -> Any:
        """
        根据种子或文件获取说话人嵌入（返回深拷贝防止被修改）
        
        Args:
            seed: 随机种子
            emb_file: 可选的 embedding 文件路径（.pt 文件）
            
        Returns:
            说话人嵌入向量的副本
        """
        # 如果指定了 embedding 文件，从文件加载
        if emb_file and HAS_TORCH:
            emb_path = Path(emb_file)
            if not emb_path.is_absolute():
                emb_path = self.assets_dir / emb_file if self.assets_dir else Path(emb_file)
            
            if emb_path.exists():
                try:
                    import torch
                    spk_emb = torch.load(str(emb_path), map_location='cpu')
                    print(f"[音色] 从文件加载: {emb_path.name}")
                    if hasattr(spk_emb, 'clone'):
                        return spk_emb.clone().detach()
                    return copy.deepcopy(spk_emb)
                except Exception as e:
                    print_warning(f"加载 embedding 文件失败: {e}，回退到种子生成")
        
        # 从种子生成
        self._set_all_seeds(seed)
        spk_emb = self.chat.sample_random_speaker()
        
        # 深拷贝，防止被 ChatTTS 内部操作修改
        if HAS_TORCH and hasattr(spk_emb, 'clone'):
            return spk_emb.clone().detach()
        return copy.deepcopy(spk_emb)
    
    def _generate_single(
        self,
        text: str,
        role: RoleConfig,
        speaker_emb: Any
    ) -> Optional[np.ndarray]:
        """
        生成单段音频
        
        Args:
            text: 文本内容
            role: 角色配置
            speaker_emb: 说话人嵌入
            
        Returns:
            音频数据，失败返回 None
        """
        # 跳过空文本或过短文本
        if not text or len(text.strip()) < 1:
            return None
        
        # 【优化】将语速控制 prompt 直接拼接到文本前，控制力更强
        # 同时在末尾加 [uv_break] 防止吞字
        text = f"{role.prompt} {text} [uv_break]"
        
        # 【核心修复】每次生成前锁定种子（必须在此处，不能省略）
        torch.manual_seed(role.seed)
        self._set_all_seeds(role.seed)
        
        # 调试：打印实际使用的角色参数
        print(f"[生成] seed={role.seed}, text={text[:20]}...")
        
        # 重试机制
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 【关键】每次 infer 调用前必须重新设置种子
                torch.manual_seed(role.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(role.seed)
                random.seed(role.seed)
                np.random.seed(role.seed)
                
                # 使用 torch.inference_mode 确保推理模式
                with torch.inference_mode():
                    # 调用 ChatTTS
                    wavs = self.chat.infer(
                        [text],
                        use_decoder=True,
                        do_text_normalization=False,
                        skip_refine_text=True,  # 关闭文本精修
                        params_infer_code=ChatTTS.Chat.InferCodeParams(
                            spk_emb=speaker_emb,  # 直接使用，ChatTTS 内部会复制
                            temperature=0.00001,  # 极低温度
                            top_P=self.top_P,
                            top_K=1,  # 贪婪解码
                            # prompt 已拼接到 text 里，这里保留作为双重保险
                            prompt=role.prompt,
                            manual_seed=role.seed,  # ChatTTS 内部种子
                        )
                    )
                
                if wavs is not None and len(wavs) > 0:
                    audio = wavs[0]
                    if audio is not None:
                        # 处理不同的返回类型
                        if hasattr(audio, 'numpy'):
                            audio = audio.numpy()
                        if isinstance(audio, np.ndarray):
                            if len(audio.shape) > 1:
                                audio = audio.flatten()
                            if len(audio) > 100:
                                return audio.astype(np.float32)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue  # 重试
                print_warning(f"生成失败: {e}")
        
        return None
    
    def generate(
        self,
        dialogues: List[DialogueItem],
        output_path: str,
        resume: bool = True,
        force: bool = False,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None
    ) -> bool:
        """
        生成完整音频
        
        Args:
            dialogues: 对话列表
            output_path: 输出文件路径
            resume: 是否从缓存恢复
            force: 是否强制重新生成
            range_start: 起始索引（可选）
            range_end: 结束索引（可选）
            
        Returns:
            是否成功
        """
        if not self.chat:
            print_error("ChatTTS 未初始化，请先调用 initialize()")
            return False
        
        if not dialogues:
            print_warning("对话列表为空")
            return False
        
        # 处理范围参数
        if range_start is not None or range_end is not None:
            start = range_start or 0
            end = range_end or len(dialogues)
            dialogues = dialogues[start:end]
            print_info(f"仅生成第 {start + 1} 到第 {end} 句")
        
        # 【关键】在生成任何音频之前，预先为所有角色生成 speaker_embedding
        # 这确保每个角色的嵌入在一开始就固定，不受后续操作影响
        speaker_embeddings: Dict[str, Any] = {}
        
        # 收集所有出现的角色
        unique_speakers = set(item.speaker_id for item in dialogues)
        print_info(f"检测到 {len(unique_speakers)} 个角色，预生成音色嵌入...")
        
        for speaker_id in sorted(unique_speakers):
            role = self.role_manager.get_role(speaker_id)
            spk_emb = self._get_speaker_embedding(role.seed, role.emb_file)
            speaker_embeddings[speaker_id] = spk_emb
            # 打印 embedding 类型和特征
            emb_type = type(spk_emb).__name__
            emb_id = id(spk_emb)
            if HAS_TORCH and hasattr(spk_emb, 'shape'):
                print(f"[角色初始化] {speaker_id}: seed={role.seed}, type={emb_type}, shape={spk_emb.shape}, id={emb_id}")
            else:
                print(f"[角色初始化] {speaker_id}: seed={role.seed}, type={emb_type}, id={emb_id}")
        
        # 存储所有音频片段
        audio_segments: List[np.ndarray] = []
        
        # 底噪文件路径
        room_tone_path = None
        if self.assets_dir:
            room_tone_file = self.assets_dir / "room_tone.wav"
            if room_tone_file.exists():
                room_tone_path = str(room_tone_file)
        
        # 计算 crossfade 样本数
        crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        
        # 进度条
        iterator = dialogues
        if HAS_TQDM:
            iterator = tqdm(dialogues, desc="生成音频", unit="句")
        
        print_info(f"开始生成，共 {len(dialogues)} 个片段...")
        
        for item in iterator:
            # 检查停止标志
            if self.stop_flag:
                print_info("用户终止生成")
                self.stop_flag = False  # 重置标志
                return False
            
            # 获取角色配置
            role = self.role_manager.get_role(item.speaker_id)
            
            # 生成缓存文件名
            cache_filename = get_cache_filename(
                item.text, item.speaker_id, role.seed, item.index
            )
            cache_path = self.temp_dir / cache_filename
            
            # 检查缓存
            audio_data = None
            
            if resume and not force and cache_path.exists():
                # 从缓存加载
                try:
                    audio_data, _ = sf.read(str(cache_path))
                    audio_data = audio_data.astype(np.float32)
                except Exception:
                    audio_data = None
            
            if audio_data is None:
                # 使用预生成的说话人嵌入（已在循环前统一生成）
                spk_emb = speaker_embeddings[item.speaker_id]
                
                # 长文本切分
                chunks = split_text(item.text, self.max_chunk_length)
                
                chunk_audios = []
                for chunk in chunks:
                    chunk_audio = self._generate_single(chunk, role, spk_emb)
                    if chunk_audio is not None:
                        chunk_audios.append(chunk_audio)
                
                if chunk_audios:
                    # 合并切分的音频
                    if len(chunk_audios) == 1:
                        audio_data = chunk_audios[0]
                    else:
                        audio_data = np.concatenate(chunk_audios)
                    
                    # 保存到缓存
                    try:
                        sf.write(str(cache_path), audio_data, self.sample_rate)
                    except Exception as e:
                        print_warning(f"保存缓存失败: {e}")
            
            if audio_data is not None:
                audio_segments.append(audio_data)
                
                # 添加停顿
                pause_audio = get_pause_audio(
                    self.pause_duration,
                    self.sample_rate,
                    self.use_room_tone,
                    room_tone_path,
                    self.room_tone_volume
                )
                audio_segments.append(pause_audio)
        
        if not audio_segments:
            print_error("没有生成任何音频")
            return False
        
        # 合并所有音频（带 crossfade）
        print_info("正在合并音频...")
        
        if crossfade_samples > 0 and len(audio_segments) > 1:
            # 使用 crossfade 合并
            full_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                full_audio = crossfade_audio(full_audio, segment, crossfade_samples)
        else:
            # 直接拼接
            full_audio = np.concatenate(audio_segments)
        
        # 应用混响 (统一空间感)
        reverb_cfg = self.config.get('audio', {}).get('reverb', {})
        if reverb_cfg.get('enabled', False):
            print_info("正在应用空间混响 (Studio Reverb)...")
            full_audio = apply_reverb(
                full_audio,
                sample_rate=self.sample_rate,
                room_size=reverb_cfg.get('room_size', 0.1),
                damping=reverb_cfg.get('damping', 0.5),
                wet_level=reverb_cfg.get('wet_level', 0.1)
            )
        
        # 音频归一化
        if self.do_normalize:
            print_info("正在归一化音频 (LUFS 标准)...")
            full_audio = normalize_audio(full_audio, sample_rate=self.sample_rate)
        
        # 验证音频数据
        if full_audio is None or len(full_audio) == 0:
            print_error("音频数据为空")
            return False
        
        # 检查并修复无效值
        if np.any(np.isnan(full_audio)) or np.any(np.isinf(full_audio)):
            print_warning("检测到无效音频数据，正在修复...")
            full_audio = np.nan_to_num(full_audio, nan=0.0, posinf=0.99, neginf=-0.99)
        
        # 确保音频在有效范围内
        max_val = np.max(np.abs(full_audio))
        if max_val > 1.0:
            full_audio = full_audio / max_val * 0.99
        
        # 保存最终音频
        print_info(f"正在保存到: {output_path}")
        
        try:
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用显式参数保存
            sf.write(output_path, full_audio.astype(np.float32), self.sample_rate, subtype='FLOAT')
            
            # 计算时长
            duration = len(full_audio) / self.sample_rate
            minutes = int(duration // 60)
            seconds = duration % 60
            
            print_success(f"音频生成完成！")
            print_info(f"文件: {output_path}")
            print_info(f"时长: {minutes}分{seconds:.1f}秒")
            
            return True
            
        except Exception as e:
            print_error(f"保存失败: {e}")
            return False
    
    def dry_run(self, dialogues: List[DialogueItem]) -> None:
        """
        空跑模式：只解析，不生成
        
        Args:
            dialogues: 对话列表
        """
        print_info("=" * 60)
        print_info("空跑模式 (Dry Run) - 仅解析，不生成音频")
        print_info("=" * 60)
        
        # 统计信息
        total_chars = 0
        speaker_stats: Dict[str, int] = {}
        
        for item in dialogues:
            role = self.role_manager.get_role(item.speaker_id)
            
            # 统计
            total_chars += len(item.text)
            speaker_stats[item.speaker_id] = speaker_stats.get(item.speaker_id, 0) + 1
            
            # 预览
            text_preview = item.text[:40]
            if len(item.text) > 40:
                text_preview += "..."
            
            print(f"[{item.index + 1:03d}] 发言人{item.speaker_id} (Seed:{role.seed}): {text_preview}")
        
        print_info("=" * 60)
        print_info("统计信息:")
        print_info(f"  总对话数: {len(dialogues)}")
        print_info(f"  总字符数: {total_chars}")
        print_info(f"  发言人数: {len(speaker_stats)}")
        
        for spk_id, count in sorted(speaker_stats.items()):
            role = self.role_manager.get_role(spk_id)
            print_info(f"    发言人{spk_id}: {count} 句 ({role.desc})")
        
        # 估算时长
        estimated_duration = total_chars * 0.3  # 粗略估计：每字 0.3 秒
        minutes = int(estimated_duration // 60)
        seconds = estimated_duration % 60
        print_info(f"  预估时长: 约 {minutes}分{seconds:.0f}秒")
        
        print_info("=" * 60)
        print_success("空跑完成！如果解析结果正确，请移除 --dry-run 参数开始生成。")

