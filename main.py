#!/usr/bin/env python3
"""
Deep Podcast - ChatTTS 多角色配音应用

使用方法:
    python main.py -i data/input/transcript.txt -o data/output/podcast.wav
    python main.py --dry-run           # 空跑模式
    python main.py --range 5-10        # 只生成第5-10句
    python main.py --resume            # 从缓存恢复
    python main.py --force             # 强制重新生成
    python main.py --clean             # 清理缓存
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[!] 配置文件不存在: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[!] 配置文件格式错误: {e}")
        sys.exit(1)


def parse_range(range_str: str) -> tuple:
    """
    解析范围参数
    
    Args:
        range_str: 范围字符串，如 "5-10" 或 "5"
        
    Returns:
        (start, end) 元组
    """
    if '-' in range_str:
        parts = range_str.split('-')
        start = int(parts[0]) - 1  # 转为 0-based
        end = int(parts[1])
        return start, end
    else:
        idx = int(range_str) - 1
        return idx, idx + 1


def clean_directories(temp_dir: str, output_dir: str):
    """清理缓存和输出目录"""
    from src.utils import clean_directory, print_info, print_success
    
    print_info("正在清理目录...")
    
    # 清理 temp 目录
    if os.path.exists(temp_dir):
        clean_directory(temp_dir, "*.wav")
        print_info(f"已清理: {temp_dir}")
    
    # 清理 output 目录
    if os.path.exists(output_dir):
        clean_directory(output_dir, "*.wav")
        print_info(f"已清理: {output_dir}")
    
    print_success("清理完成！")


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 默认路径
    default_input = str(project_root / "data" / "input" / "transcript.txt")
    default_output = str(project_root / "data" / "output" / "podcast.wav")
    default_config = str(project_root / "config.yaml")
    default_temp = str(project_root / "temp")
    default_assets = str(project_root / "assets")
    
    # 命令行参数
    parser = argparse.ArgumentParser(
        description="Deep Podcast - ChatTTS 多角色配音应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 使用默认路径
  python main.py -i script.txt -o output.wav        # 指定输入输出
  python main.py --dry-run                          # 空跑模式（只解析不生成）
  python main.py --range 5-10                       # 只生成第5-10句
  python main.py --resume                           # 从缓存恢复
  python main.py --force                            # 忽略缓存，强制重生成
  python main.py --clean                            # 清空缓存目录
        """
    )
    
    # 基本参数
    parser.add_argument(
        '-i', '--input',
        default=default_input,
        help=f'输入文件路径 (默认: {default_input})'
    )
    parser.add_argument(
        '-o', '--output',
        default=default_output,
        help=f'输出文件路径 (默认: {default_output})'
    )
    parser.add_argument(
        '-c', '--config',
        default=default_config,
        help=f'配置文件路径 (默认: {default_config})'
    )
    
    # 调试参数
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='空跑模式：只解析文档，不生成音频'
    )
    parser.add_argument(
        '--range',
        type=str,
        help='指定生成范围，如 "5-10" 或 "5"'
    )
    
    # 缓存控制
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='从缓存恢复（默认启用）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成，忽略缓存'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='清空 temp/ 和 output/ 目录'
    )
    
    args = parser.parse_args()
    
    # 清理模式
    if args.clean:
        clean_directories(default_temp, str(project_root / "data" / "output"))
        return
    
    # 导入模块（延迟导入，加快启动速度）
    from src.utils import print_info, print_success, print_error, print_warning
    from src.parser import parse_transcript, format_dialogue_preview
    from src.role_manager import RoleManager
    from src.generator import AudioGenerator
    
    # 打印 Banner
    print()
    print("=" * 60)
    print("  Deep Podcast - ChatTTS 多角色配音应用")
    print("=" * 60)
    print()
    
    # 加载配置
    print_info(f"加载配置: {args.config}")
    config = load_config(args.config)
    
    # 初始化角色管理器
    role_manager = RoleManager(config)
    
    # 解析文档
    dialogues = parse_transcript(
        args.input,
        normalize_text=True,
        convert_numbers=config.get('text', {}).get('convert_numbers', True)
    )
    
    if not dialogues:
        print_error("文档解析失败或内容为空")
        sys.exit(1)
    
    # 处理范围参数
    range_start = None
    range_end = None
    if args.range:
        try:
            range_start, range_end = parse_range(args.range)
            print_info(f"指定范围: 第 {range_start + 1} 到第 {range_end} 句")
        except ValueError:
            print_error(f"无效的范围格式: {args.range}")
            sys.exit(1)
    
    # 初始化生成器
    generator = AudioGenerator(config, role_manager)
    
    # 空跑模式
    if args.dry_run:
        # 先预览角色配置
        print()
        print(role_manager.summary())
        print()
        
        # 执行空跑
        if range_start is not None:
            generator.dry_run(dialogues[range_start:range_end])
        else:
            generator.dry_run(dialogues)
        return
    
    # 初始化 ChatTTS
    if not generator.initialize(temp_dir=default_temp, assets_dir=default_assets):
        print_error("模型初始化失败")
        sys.exit(1)
    
    # 打印角色配置
    print()
    print(role_manager.summary())
    print()
    
    # 生成音频
    success = generator.generate(
        dialogues,
        args.output,
        resume=args.resume,
        force=args.force,
        range_start=range_start,
        range_end=range_end
    )
    
    if not success:
        print_error("音频生成失败")
        sys.exit(1)
    
    print()
    print_success("全部完成！")


if __name__ == "__main__":
    main()

