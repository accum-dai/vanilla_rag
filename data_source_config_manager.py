import os
import yaml
import json
import datetime
from typing import Dict, List, Any, Optional, Union
import pathlib


class DataSourceConfigManager:
    """
    管理数据源配置的类，支持YAML和JSON格式
    """
    def __init__(self, config_path: str, data_sources_dir: str = "data_sources"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径 (.yaml或.json)
            data_sources_dir: 数据源配置文件存放目录
        """
        # 创建数据源目录（如果不存在）
        self.data_sources_dir = data_sources_dir
        if not os.path.exists(self.data_sources_dir):
            os.makedirs(self.data_sources_dir, exist_ok=True)
            print(f"创建数据源配置目录: {self.data_sources_dir}")
        
        # 直接使用提供的配置文件路径，不进行任何组合
        self.config_path = config_path
        
        print(f"配置文件路径: {self.config_path}")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            print(f"配置文件不存在: {self.config_path}")
            return {"data_sources": []}
        
        file_ext = os.path.splitext(self.config_path)[1].lower()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    config = yaml.safe_load(f) or {"data_sources": []}
                    print(f"成功加载YAML配置: {self.config_path}")
                    print(f"包含 {len(config.get('data_sources', []))} 个数据源")
                    return config
                elif file_ext == '.json':
                    config = json.load(f) or {"data_sources": []}
                    print(f"成功加载JSON配置: {self.config_path}")
                    print(f"包含 {len(config.get('data_sources', []))} 个数据源")
                    return config
                else:
                    print(f"不支持的配置文件格式: {file_ext}")
                    return {"data_sources": []}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {"data_sources": []}
    
    def save_config(self, backup: bool = True) -> str:
        """
        保存配置文件，可选择创建备份
        
        Args:
            backup: 是否创建带时间戳的备份
            
        Returns:
            保存的文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
        
        save_path = self.config_path
        
        # 如果需要备份，生成带时间戳的文件名
        if backup:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(self.config_path)
            base_name, ext = os.path.splitext(filename)
            backup_filename = f"{base_name}_{timestamp}{ext}"
            save_path = os.path.join(self.data_sources_dir, backup_filename)
        
        file_ext = os.path.splitext(save_path)[1].lower()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                elif file_ext == '.json':
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                else:
                    print(f"不支持的配置文件格式: {file_ext}")
            
            print(f"配置保存到: {save_path}")
            return save_path
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return ""
    
    def get_data_sources(self) -> List[Dict[str, Any]]:
        """获取所有数据源配置"""
        return self.config.get("data_sources", [])
    
    def get_unprocessed_data_sources(self) -> List[Dict[str, Any]]:
        """获取未处理的数据源配置"""
        return [ds for ds in self.get_data_sources() if ds.get("status") != "processed"]
    
    def update_data_source_status(self, source_path: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新数据源状态
        
        Args:
            source_path: 数据源路径
            status: 新状态 (如 "processed")
            metadata: 要更新的元数据
            
        Returns:
            是否成功更新
        """
        for ds in self.config.get("data_sources", []):
            if ds["source_path"] == source_path:
                ds["status"] = status
                ds["last_updated"] = datetime.datetime.now().isoformat()
                
                if metadata:
                    if "metadata" not in ds:
                        ds["metadata"] = {}
                    ds["metadata"].update(metadata)
                
                return True
        
        return False