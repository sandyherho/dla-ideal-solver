"""Configuration file parser for DLA simulations."""

from pathlib import Path


class ConfigManager:
    """Parse configuration files for DLA simulations."""
    
    @staticmethod
    def load(config_path: str) -> dict:
        """Load configuration from file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = {}
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                value = ConfigManager._parse_value(value)
                config[key] = value
        
        return config
    
    @staticmethod
    def _parse_value(value: str):
        """Parse string value to appropriate Python type."""
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
