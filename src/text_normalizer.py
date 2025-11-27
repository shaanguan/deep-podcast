"""
文本预处理模块
- 数字转汉字（使用 cn2an smart 模式）
- 保留 ChatTTS 原生标签
- 清理特殊符号
"""

import re
from typing import Optional

try:
    import cn2an
    HAS_CN2AN = True
except ImportError:
    HAS_CN2AN = False


class TextNormalizer:
    """文本标准化处理器"""
    
    # ChatTTS 原生标签，清洗时需要保留
    CHATTTS_TAGS = [
        r'\[laugh\]',
        r'\[uv_break\]',
        r'\[break_\d+\]',
        r'\[speed_\d+\]',
        r'\[oral_\d+\]',
        r'\[laugh_\d+\]',
    ]
    
    # 需要清理的特殊字符（可能导致模型不稳定）
    SPECIAL_CHARS = [
        '【', '】', '《', '》', '〈', '〉',
        '「', '」', '『', '』',
        '★', '☆', '●', '○', '◆', '◇',
        '▲', '△', '▼', '▽',
        '■', '□', '◎', '⊙',
        '→', '←', '↑', '↓', '↔',
        '※', '§', '¶', '†', '‡',
    ]
    
    def __init__(self, convert_numbers: bool = True):
        """
        初始化文本标准化器
        
        Args:
            convert_numbers: 是否将数字转换为汉字
        """
        self.convert_numbers = convert_numbers
        
        # 构建标签保护正则
        self._tag_pattern = re.compile(
            '(' + '|'.join(self.CHATTTS_TAGS) + ')',
            re.IGNORECASE
        )
        
        # 构建特殊字符清理正则
        escaped_chars = [re.escape(c) for c in self.SPECIAL_CHARS]
        self._special_char_pattern = re.compile(
            '[' + ''.join(escaped_chars) + ']'
        )
    
    def normalize(self, text: str) -> str:
        """
        对文本进行标准化处理
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return text
        
        # Step 1: 保护 ChatTTS 标签（用占位符替换）
        protected_tags = []
        def protect_tag(match):
            tag = match.group(0)
            placeholder = f"__TAG_{len(protected_tags)}__"
            protected_tags.append(tag)
            return placeholder
        
        text = self._tag_pattern.sub(protect_tag, text)
        
        # Step 2: 数字转汉字
        if self.convert_numbers and HAS_CN2AN:
            text = self._convert_numbers(text)
        
        # Step 3: 清理特殊字符
        text = self._clean_special_chars(text)
        
        # Step 4: 规范化标点符号
        text = self._normalize_punctuation(text)
        
        # Step 5: 恢复保护的标签
        for i, tag in enumerate(protected_tags):
            text = text.replace(f"__TAG_{i}__", tag)
        
        # Step 6: 清理多余空白
        text = self._clean_whitespace(text)
        
        return text
    
    def _convert_numbers(self, text: str) -> str:
        """
        将阿拉伯数字转换为汉字
        """
        if not HAS_CN2AN:
            # 手动转换基本数字
            return self._manual_number_convert(text)
        
        try:
            # 使用 cn2an 的 smart 模式
            result = cn2an.transform(text, "smart")
            # 如果还有残留数字，手动处理
            if re.search(r'\d', result):
                result = self._manual_number_convert(result)
            return result
        except Exception:
            # 转换失败时使用手动转换
            return self._manual_number_convert(text)
    
    def _manual_number_convert(self, text: str) -> str:
        """手动转换数字为汉字"""
        digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        for digit, hanzi in digit_map.items():
            text = text.replace(digit, hanzi)
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """清理特殊字符"""
        # 将特殊字符替换为空格
        text = self._special_char_pattern.sub(' ', text)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号，确保 ChatTTS 兼容"""
        # ChatTTS 支持的标点符号有限，需要转换或移除不支持的
        replacements = {
            # 中文标点转换
            '，': ',',
            '。': '.',
            '！': ',',   # ChatTTS 不支持 !，用逗号替代
            '？': ',',   # ChatTTS 不支持 ?，用逗号替代（语气通过 refine 控制）
            '；': ',',
            '：': ',',
            '"': '',
            '"': '',
            ''': '',
            ''': '',
            '（': ',',
            '）': ',',
            '【': '',
            '】': '',
            '《': '',
            '》': '',
            # 全角符号
            '－': '',    # 全角减号
            '—': ',',    # 破折号
            '…': ',',    # 省略号
            '～': '',    # 波浪号
            # 英文标点也要处理
            '!': ',',
            '?': ',',
            ';': ',',
            ':': ',',
            '"': '',
            "'": '',
            '(': ',',
            ')': ',',
            '-': '',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 移除连续的逗号和句号
        text = re.sub(r',+', ',', text)
        text = re.sub(r'\.+', '.', text)
        
        # 移除开头和结尾的逗号
        text = text.strip(',. ')
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """清理多余空白"""
        # 将多个空格合并为一个
        text = re.sub(r' +', ' ', text)
        # 移除首尾空白
        text = text.strip()
        return text


def normalize(text: str, convert_numbers: bool = True) -> str:
    """
    便捷函数：对文本进行标准化处理
    
    Args:
        text: 原始文本
        convert_numbers: 是否将数字转换为汉字
        
    Returns:
        处理后的文本
    """
    normalizer = TextNormalizer(convert_numbers=convert_numbers)
    return normalizer.normalize(text)

