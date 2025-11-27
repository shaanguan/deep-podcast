"""
文档解析器模块
- 解析 transcript.txt 文件
- 提取发言人 ID 和内容
- 支持多种格式
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .utils import print_info, print_warning, print_error


@dataclass
class DialogueItem:
    """对话条目"""
    index: int              # 序号（从 0 开始）
    speaker_id: str         # 发言人 ID
    text: str               # 文本内容
    raw_text: str           # 原始文本（未处理）


def parse_transcript(
    file_path: str,
    normalize_text: bool = True,
    convert_numbers: bool = True
) -> List[DialogueItem]:
    """
    解析对话文档
    
    支持的格式：
    1. "发言人1 内容" （空格分隔）
    2. "发言人1: 内容" （冒号分隔）
    3. "发言人1\n内容" （换行分隔）
    
    Args:
        file_path: 文件路径
        normalize_text: 是否进行文本标准化
        convert_numbers: 是否将数字转汉字
        
    Returns:
        对话条目列表
    """
    print_info(f"正在解析文档: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print_error(f"找不到文件: {file_path}")
        return []
    except Exception as e:
        print_error(f"读取文件失败: {e}")
        return []
    
    lines = content.split('\n')
    
    dialogues: List[DialogueItem] = []
    current_speaker_id: Optional[str] = None
    current_text_parts: List[str] = []
    
    # 正则模式：匹配发言人标记
    # 支持: "发言人1", "发言人 1", "Speaker 1", "说话人1" 等
    speaker_pattern = re.compile(
        r'^\s*(发言人|说话人|讲者|Speaker)\s*(\d+)\s*[:：]?\s*(.*)$',
        re.IGNORECASE
    )
    
    # 清理标签的模式
    source_tag_pattern = re.compile(r'\s*<[^>]+>\s*')
    
    # 时间戳模式（需要在解析前移除）
    # 支持: "00:15:30", "[00:15:30]", "(00:15)", "00:15" 等
    timestamp_pattern = re.compile(
        r'[\[\(]?\d{1,2}:\d{2}(?::\d{2})?[\]\)]?\s*'
    )
    
    def save_current_dialogue():
        """保存当前积累的对话"""
        nonlocal current_speaker_id, current_text_parts
        
        if current_speaker_id is not None and current_text_parts:
            # 合并文本，检测段落间的换行来插入停顿标签
            # 如果 parts 之间有空行（表示换段落），插入停顿标签
            merged_parts = []
            for i, part in enumerate(current_text_parts):
                merged_parts.append(part)
                # 检查是否是段落分隔（空字符串表示原文有空行）
                if part == '__PARAGRAPH_BREAK__':
                    merged_parts[-1] = '[uv_break]'  # 用停顿标签替换
            
            raw_text = ' '.join(p for p in merged_parts if p != '__PARAGRAPH_BREAK__')
            
            # 清理 HTML/XML 标签
            clean_text = source_tag_pattern.sub('', raw_text)
            clean_text = clean_text.strip()
            
            if clean_text:
                # 文本标准化
                if normalize_text:
                    from .text_normalizer import normalize
                    processed_text = normalize(clean_text, convert_numbers)
                else:
                    processed_text = clean_text
                
                dialogues.append(DialogueItem(
                    index=len(dialogues),
                    speaker_id=current_speaker_id,
                    text=processed_text,
                    raw_text=raw_text
                ))
        
        current_speaker_id = None
        current_text_parts = []
    
    prev_was_empty = False  # 跟踪上一行是否为空行
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # 检测空行（可能表示段落分隔，需要插入停顿）
        if not line:
            if current_speaker_id is not None and current_text_parts:
                prev_was_empty = True
            continue
        
        # 移除行首的时间戳（如 "00:15:30" 或 "[00:15]"）
        line = timestamp_pattern.sub('', line).strip()
        
        # 移除时间戳后如果为空，跳过
        if not line:
            continue
        
        # 尝试匹配发言人标记
        match = speaker_pattern.match(line)
        
        if match:
            # 保存上一段对话
            save_current_dialogue()
            prev_was_empty = False
            
            # 开始新的对话
            current_speaker_id = match.group(2)
            
            # 如果同一行有内容，添加到文本
            remaining_text = match.group(3).strip()
            if remaining_text:
                current_text_parts.append(remaining_text)
        else:
            # 不是发言人标记，作为内容追加
            if current_speaker_id is not None:
                # 如果之前有空行，先插入段落分隔标记
                if prev_was_empty:
                    current_text_parts.append('__PARAGRAPH_BREAK__')
                    prev_was_empty = False
                current_text_parts.append(line)
            else:
                # 如果还没有发言人，跳过这些内容
                # （可能是文档开头的注释或说明）
                pass
    
    # 保存最后一段对话
    save_current_dialogue()
    
    print_info(f"解析完成，共 {len(dialogues)} 个对话片段")
    
    # 统计发言人
    speaker_ids = set(d.speaker_id for d in dialogues)
    print_info(f"发现 {len(speaker_ids)} 位发言人: {', '.join(sorted(speaker_ids))}")
    
    return dialogues


def format_dialogue_preview(
    dialogues: List[DialogueItem],
    max_text_length: int = 30
) -> str:
    """
    格式化对话预览
    
    Args:
        dialogues: 对话列表
        max_text_length: 文本最大显示长度
        
    Returns:
        预览字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("对话预览")
    lines.append("=" * 60)
    
    for d in dialogues:
        text_preview = d.text[:max_text_length]
        if len(d.text) > max_text_length:
            text_preview += "..."
        
        lines.append(f"[{d.index + 1:03d}] 发言人{d.speaker_id}: {text_preview}")
    
    lines.append("=" * 60)
    return "\n".join(lines)

