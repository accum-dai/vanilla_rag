import os
import yaml
import dotenv
from typing import Dict, Any, Optional, Union


class AppConfigManager:
    """管理应用程序配置的类，支持YAML和.env格式的配置混合使用"""
    
    def __init__(self, config_path: str = "app_config.yaml"):
        """
        初始化应用程序配置管理器
        
        Args:
            config_path: YAML配置文件路径
        """
        self.config_path = config_path
        
        # 加载.env文件（如果存在）
        dotenv.load_dotenv()
        
        # 加载YAML配置
        self.config = self._load_yaml_config()
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Returns:
            配置字典
        """
        if not os.path.exists(self.config_path):
            print(f"警告: 配置文件不存在: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            print(f"已加载配置文件: {self.config_path}")
            
            # 添加调试信息，打印配置结构
            print("配置结构:")
            for key in config:
                if isinstance(config[key], dict):
                    print(f"  {key}:")
                    for subkey in config[key]:
                        print(f"    {subkey}")
                        if isinstance(config[key][subkey], dict):
                            for subsubkey in config[key][subkey]:
                                print(f"      {subsubkey}: {config[key][subkey][subsubkey]}")
                else:
                    print(f"  {key}: {config[key]}")
            
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖YAML配置"""
        # 示例映射，可以根据需要扩展
        env_mapping = {
            "OLLAMA_API_URL": ["generator", "ollama", "api_url"],
            "OLLAMA_API_KEY": ["generator", "ollama", "api_key"],
            "OLLAMA_DEFAULT_MODEL": ["generator", "ollama", "default_model"],
            
            "OPENAI_API_URL": ["generator", "openai", "api_url"],
            "OPENAI_API_KEY": ["generator", "openai", "api_key"],
            "OPENAI_DEFAULT_MODEL": ["generator", "openai", "default_model"],
            
            "EMBEDDING_MODEL_NAME": ["embedding", "sentence_transformer", "model_name"],
            "VECTOR_STORE_PERSIST_DIR": ["vector_store", "persist_dir"],
            "VECTOR_STORE_DEFAULT_COLLECTION": ["vector_store", "default_collection"],
        }
        
        for env_var, config_path in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                self._set_nested_value(self.config, config_path, env_value)
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any) -> None:
        """
        在嵌套字典中设置值
        
        Args:
            config: 配置字典
            path: 路径列表
            value: 要设置的值
        """
        current = config
        for i, key in enumerate(path):
            if i == len(path) - 1:
                current[key] = value
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
    
    def get(self, *path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径（支持多级嵌套）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        current = self.config
        
        # 添加调试信息 - 修复: 将所有路径元素转换为字符串
        path_str = " -> ".join(str(p) for p in path)
        print(f"尝试获取配置: {path_str}")
        
        for key in path:
            if not isinstance(current, dict):
                print(f"  配置路径错误: {path_str} (当前节点不是字典)")
                return default
            
            if key not in current:
                print(f"  配置键不存在: {key} in {path_str}")
                return default
            
            current = current[key]
        
        # 如果值为None，返回默认值
        if current is None:
            print(f"  配置值为None: {path_str}, 使用默认值: {default}")
            return default
        
        print(f"  找到配置值: {path_str} = {current}")
        return current
    
    def get_generator_config(self, generator_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取生成器配置
        
        Args:
            generator_type: 生成器类型，如果为None则使用默认类型
            
        Returns:
            生成器配置字典
        """
        if generator_type is None:
            generator_type = self.get("generator", "default_type", default="ollama")
        
        return self.get("generator", generator_type, default={})
    
    def get_embedding_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        获取嵌入模型配置
        
        Args:
            provider: 提供商，如果为None则使用默认提供商
            
        Returns:
            嵌入模型配置字典
        """
        if provider is None:
            provider = self.get("embedding", "default_provider", default="sentence_transformer")
        
        return self.get("embedding", provider, default={})
    
    def save(self) -> bool:
        """
        保存配置到YAML文件
        
        Returns:
            是否成功保存
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存到: {self.config_path}")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False