import yaml
import re
from typing import Any, Dict
from pathlib import Path

class UnresolvedVariableError(Exception):
    """Custom exception for unresolved variables."""
    pass

class CircularDependencyError(Exception):
    """Custom exception for circular dependencies."""
    pass

class KinectConfigFileHandler:
    """
    Handles the loading and resolution of YAML configuration files with variable placeholders.
    """
    
    @staticmethod
    def _flatten_dict_for_resolution(data: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
        """
        Recursively flattens a dictionary to get all string values that can be used as variables.
        """
        items = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(KinectConfigFileHandler._flatten_dict_for_resolution(value, new_key))
            elif isinstance(value, str):
                items[key] = value
        return items

    @staticmethod
    def _recursive_substitute(data: Any, resolved_vars: Dict[str, str]) -> Any:
        """
        Recursively substitutes placeholders in a data structure with their resolved values.
        """
        if isinstance(data, dict):
            return {k: KinectConfigFileHandler._recursive_substitute(v, resolved_vars) for k, v in data.items()}
        elif isinstance(data, list):
            return [KinectConfigFileHandler._recursive_substitute(item, resolved_vars) for item in data]
        elif isinstance(data, str):
            for _ in range(len(resolved_vars)): 
                for placeholder, value in resolved_vars.items():
                    data = data.replace(f"{{{placeholder}}}", value)
            return data
        else:
            return data

    @staticmethod
    def load_and_resolve_config(filepath: str) -> Dict[str, Any]:
        """
        Loads a YAML file, resolves all internal placeholders, and returns the result.

        Args:
            filepath (str): The path to the YAML configuration file.

        Returns:
            Dict[str, Any]: The fully resolved configuration.
            
        Raises:
            FileNotFoundError: If the filepath does not exist.
            CircularDependencyError: If a circular dependency is detected.
            UnresolvedVariableError: If a variable cannot be resolved.
        """
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        variables = KinectConfigFileHandler._flatten_dict_for_resolution(config_data)
        
        max_iterations = len(variables) + 1 
        for i in range(max_iterations):
            changed_in_pass = False
            for key, value in variables.items():
                placeholders = re.findall(r'\{([^}]+)\}', value)
                if not placeholders:
                    continue

                for placeholder in placeholders:
                    if placeholder in variables:
                        sub_value = variables[placeholder]
                        if '{' not in sub_value:
                            new_value = value.replace(f"{{{placeholder}}}", sub_value)
                            if new_value != value:
                                variables[key] = new_value
                                value = new_value
                                changed_in_pass = True
            
            if not changed_in_pass:
                break
        
        if i == max_iterations - 1:
            raise CircularDependencyError("A circular dependency was detected in the configuration file.")

        final_unresolved = [
            p for val in variables.values() 
            for p in re.findall(r'\{([^}]+)\}', val) 
            if p in variables
        ]
        
        if final_unresolved:
            raise UnresolvedVariableError(f"Could not resolve variables: {set(final_unresolved)}")

        return KinectConfigFileHandler._recursive_substitute(config_data, variables)
    

def get_block_files(kinect_configs_dir: Path):
    """Helper to find and validate session configuration files."""
    if not kinect_configs_dir.is_dir():
        raise ValueError(f"Sessions folder not found: {kinect_configs_dir}")
    files = list(kinect_configs_dir.glob("*.yaml"))
    print(f"Found {len(files)} session(s) to process.")
    return files