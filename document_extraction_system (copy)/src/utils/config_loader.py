# src/utils/config_loader.py
"""
Configuration loader for document templates
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage document configuration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "document_templates.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_document_config(self, document_type: str) -> Dict[str, Any]:
        """Get configuration for a specific document type"""
        if document_type not in self.config:
            raise ValueError(f"Unknown document type: {document_type}")
        
        return self.config[document_type]
    
    def get_required_fields(self, document_type: str) -> List[str]:
        """Get required fields for a document type"""
        config = self.get_document_config(document_type)
        return config.get('required_fields', [])
    
    def get_optional_fields(self, document_type: str) -> List[str]:
        """Get optional fields for a document type"""
        config = self.get_document_config(document_type)
        return config.get('optional_fields', [])
    
    def get_key_indicators(self, document_type: str) -> List[str]:
        """Get key indicators for document type detection"""
        config = self.get_document_config(document_type)
        return config.get('key_indicators', [])
    
    def save_config(self, config: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """Save configuration to YAML file"""
        output_path = output_path or self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)