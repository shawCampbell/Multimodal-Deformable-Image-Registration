import yaml
"""
Return correct config 

Attributes:
    config (yaml object): yaml file object
"""
class Config:
    def __init__(self, file_path='C:\\dev_personal\\Thesis\\config_nfiti.yaml'):
        self.file_path = file_path
        self.config_yaml = self.load_config(self.file_path)
        self.train_cache = {}
        self.test_cache = {}

    def load_config(self, file_path):
        """Safely loads a YAML configuration file."""
        with open(file_path, "r") as stream:
            try:
                # safe_load prevents arbitrary code execution vulnerabilities
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")