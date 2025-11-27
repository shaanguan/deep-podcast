# Deep Podcast - ChatTTS 多角色配音应用
# src 模块

from .parser import parse_transcript
from .role_manager import RoleManager
from .generator import AudioGenerator
from .text_normalizer import TextNormalizer
from .utils import (
    check_hardware,
    get_device,
    generate_room_tone,
    load_room_tone,
    crossfade_audio,
    normalize_audio,
    split_text
)

__version__ = "1.0.0"

