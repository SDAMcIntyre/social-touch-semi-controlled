import yaml
import re
from typing import Any, Dict, List

class UnresolvedVariableError(Exception):
    """Custom exception for unresolved variables."""
    pass

class CircularDependencyError(Exception):
    """Custom exception for circular dependencies."""
    pass

def _flatten_dict_for_resolution(data: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
    """
    Recursively flattens a dictionary to get all string values that can be used as variables.
    It creates unique keys for nested values to avoid collisions.
    """
    items = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict_for_resolution(value, new_key))
        elif isinstance(value, str):
            # We only care about strings for variable replacement.
            items[key] = value
    return items


def _recursive_substitute(data: Any, resolved_vars: Dict[str, str]) -> Any:
    """
    Recursively traverses the original data structure and substitutes placeholders
    with their fully resolved values.
    """
    if isinstance(data, dict):
        return {k: _recursive_substitute(v, resolved_vars) for k, v in data.items()}
    elif isinstance(data, list):
        return [_recursive_substitute(item, resolved_vars) for item in data]
    elif isinstance(data, str):
        # Using a custom substitution loop to handle unresolved placeholders gracefully
        # (e.g., {session_id}, {block_id}) which are not part of the config itself.
        for _ in range(len(resolved_vars)): # Loop to handle nested resolutions
             for placeholder, value in resolved_vars.items():
                 data = data.replace(f"{{{placeholder}}}", value)
        return data
    else:
        # Return non-string, non-collection types as-is (e.g., int, bool).
        return data


def load_and_resolve_config(filepath: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file, resolves all internal variable placeholders,
    and returns the fully resolved configuration as a dictionary.

    Args:
        filepath (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration with all internal paths resolved.
        
    Raises:
        FileNotFoundError: If the specified filepath does not exist.
        CircularDependencyError: If a circular dependency is detected.
        UnresolvedVariableError: If a variable cannot be resolved after max iterations.
    """
    # 1. Load the YAML file
    with open(filepath, 'r') as f:
        config_data = yaml.safe_load(f)

    # 2. Create a flat dictionary of all potential variables.
    variables = _flatten_dict_for_resolution(config_data)
    
    # 3. Iteratively resolve placeholders.
    # We set a max iteration count to prevent infinite loops from circular dependencies.
    max_iterations = len(variables) + 1 
    for i in range(max_iterations):
        changed_in_pass = False
        unresolved_found_in_pass = False
        
        for key, value in variables.items():
            # Find all placeholders like {var} in the current string value.
            placeholders_in_value = re.findall(r'\{([^}]+)\}', value)
            
            if not placeholders_in_value:
                continue

            # Attempt to substitute each placeholder.
            for placeholder in placeholders_in_value:
                if placeholder in variables:
                    sub_value = variables[placeholder]
                    # CRITICAL: Only substitute if the placeholder's value is itself fully resolved.
                    if '{' not in sub_value:
                        new_value = value.replace(f"{{{placeholder}}}", sub_value)
                        if new_value != value:
                            variables[key] = new_value
                            value = new_value # Update for subsequent replacements in the same string
                            changed_in_pass = True
                # Note: We ignore placeholders like {session_id} which are not keys in our config.
        
        # If a full pass makes no changes, we are done.
        if not changed_in_pass:
            break
    
    # Check for errors after the loop
    if i == max_iterations - 1:
        # If we hit max iterations, it's likely a circular dependency.
        raise CircularDependencyError("A circular dependency was detected in the configuration file.")

    # 4. Final check for any variables that could not be resolved.
    final_unresolved = []
    for key, value in variables.items():
        placeholders_in_value = re.findall(r'\{([^}]+)\}', value)
        for placeholder in placeholders_in_value:
            if placeholder in variables: # It's a config var, not a dynamic one
                final_unresolved.append(placeholder)
    
    if final_unresolved:
        raise UnresolvedVariableError(f"Could not resolve variables: {set(final_unresolved)}")


    # 5. Recursively substitute the resolved values back into the original structure.
    resolved_config = _recursive_substitute(config_data, variables)

    return resolved_config