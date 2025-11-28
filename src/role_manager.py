"""
角色管理器模块
- 从配置文件加载固定角色
- 自动分配新角色的声纹种子
- 管理演技参数（refine 配置）
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RoleConfig:
    """角色配置数据类"""
    seed: int
    prompt: str = "[speed_5]"
    desc: str = ""
    
    # refine 参数
    oral: int = 2
    laugh: int = 0
    break_level: int = 4  # 'break' 是 Python 保留字
    
    # 是否为自动生成的角色
    is_auto_generated: bool = False


class RoleManager:
    """角色管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化角色管理器
        
        Args:
            config: 完整配置字典（从 config.yaml 加载）
        """
        self.config = config
        self.role_cache: Dict[str, RoleConfig] = {}
        
        # 读取自动种子配置
        auto_seed_config = config.get('auto_seed', {})
        self.auto_seed_base = auto_seed_config.get('base', 5000)
        self.auto_seed_step = auto_seed_config.get('step', 337)
        
        # 读取全局演技配置
        style_config = config.get('style', {})
        refine_config = style_config.get('refine', {})
        
        self.default_oral = refine_config.get('oral', 2)
        self.default_laugh = refine_config.get('laugh', 0)
        self.default_break = refine_config.get('break', 4)
        self.default_prompt = "[speed_5]"
        
        # 预加载固定角色
        self._load_fixed_roles()
    
    def _load_fixed_roles(self):
        """从配置加载固定角色"""
        roles_config = self.config.get('roles', {})
        
        for speaker_id, role_data in roles_config.items():
            speaker_id = str(speaker_id)  # 确保是字符串
            
            # 读取角色特定的 refine 覆盖
            refine_override = role_data.get('refine_override', {})
            
            role = RoleConfig(
                seed=role_data.get('seed', self._generate_seed(speaker_id)),
                prompt=role_data.get('prompt', self.default_prompt),
                desc=role_data.get('desc', f"角色{speaker_id}"),
                oral=refine_override.get('oral', self.default_oral),
                laugh=refine_override.get('laugh', self.default_laugh),
                break_level=refine_override.get('break', self.default_break),
                is_auto_generated=False
            )
            
            self.role_cache[speaker_id] = role
    
    def _generate_seed(self, speaker_id: str) -> int:
        """
        根据发言人 ID 生成确定性的种子值
        
        Args:
            speaker_id: 发言人 ID
            
        Returns:
            种子值
        """
        try:
            id_num = int(speaker_id)
        except ValueError:
            # 如果 ID 不是数字，使用字符串哈希
            id_num = abs(hash(speaker_id)) % 1000
        
        return self.auto_seed_base + (id_num * self.auto_seed_step)
    
    def get_role(self, speaker_id: str) -> RoleConfig:
        """
        获取角色配置，如果不存在则自动创建
        
        Args:
            speaker_id: 发言人 ID
            
        Returns:
            角色配置
        """
        speaker_id = str(speaker_id)
        
        # 已缓存的角色
        if speaker_id in self.role_cache:
            return self.role_cache[speaker_id]
        
        # 自动生成新角色
        seed = self._generate_seed(speaker_id)
        
        role = RoleConfig(
            seed=seed,
            prompt=self.default_prompt,
            desc=f"自动生成角色(ID:{speaker_id})",
            oral=self.default_oral,
            laugh=self.default_laugh,
            break_level=self.default_break,
            is_auto_generated=True
        )
        
        self.role_cache[speaker_id] = role
        
        from .utils import print_info
        print_info(f"发现新角色: 发言人{speaker_id} -> 分配 Seed: {seed}")
        
        return role
    
    def get_refine_prompt(self, role: RoleConfig) -> str:
        """
        生成 ChatTTS RefineText 的 prompt 字符串
        
        Args:
            role: 角色配置
            
        Returns:
            refine prompt 字符串
        """
        return f"[oral_{role.oral}][laugh_{role.laugh}][break_{role.break_level}]"
    
    def list_roles(self) -> Dict[str, RoleConfig]:
        """
        列出所有已知角色
        
        Returns:
            角色配置字典
        """
        return self.role_cache.copy()
    
    def summary(self) -> str:
        """
        生成角色摘要信息
        
        Returns:
            摘要字符串
        """
        lines = ["角色配置摘要:"]
        lines.append("-" * 50)
        
        for speaker_id, role in sorted(self.role_cache.items()):
            status = "自动" if role.is_auto_generated else "固定"
            lines.append(
                f"  发言人{speaker_id}: {role.desc} "
                f"(Seed: {role.seed}, {status})"
            )
        
        lines.append("-" * 50)
        return "\n".join(lines)

