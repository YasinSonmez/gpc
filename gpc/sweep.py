"""Hyperparameter sweep utilities for GPC experiments."""
import itertools
from pathlib import Path
from typing import Any

import yaml

from gpc.config import TrainingConfig


def expand_config_sweep(config_path: str | Path) -> list[tuple[TrainingConfig, str]]:
    """Expand a config file with list values into multiple configs.
    
    Any parameter that is a list will be treated as a hyperparameter to sweep over.
    All combinations of list parameters will be generated.
    
    Args:
        config_path: Path to YAML config file with potential list values.
        
    Returns:
        List of (config, experiment_suffix) tuples where:
        - config: TrainingConfig with single values
        - experiment_suffix: String describing the hyperparameter combination
        
    Example:
        If config has:
            num_iters: [10, 20]
            learning_rate: [0.001, 0.01]
            
        This will generate 4 configs with suffixes:
            - "iters10_lr0.001"
            - "iters10_lr0.01"
            - "iters20_lr0.001"
            - "iters20_lr0.01"
    """
    # Load raw YAML data
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Find all list parameters
    sweep_params = {}
    fixed_params = {}
    
    for key, value in data.items():
        # Nested lists indicate a sweep over list values
        # E.g., hidden_layers: [[32, 32], [64, 64]] sweeps over two architectures
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            # It's a sweep parameter over list values
            sweep_params[key] = value
        elif isinstance(value, list) and not _is_structural_list(key):
            # It's a sweep parameter over scalar values
            sweep_params[key] = value
        else:
            fixed_params[key] = value
    
    # If no sweep parameters, return single config
    if not sweep_params:
        config = TrainingConfig(**data)
        suffix = ""
        return [(config, suffix)]
    
    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(itertools.product(*param_values))
    
    configs = []
    for combo in combinations:
        # Create config dict for this combination
        config_dict = fixed_params.copy()
        suffix_parts = []
        
        for param_name, param_value in zip(param_names, combo):
            config_dict[param_name] = param_value
            
            # Create short suffix name
            short_name = _shorten_param_name(param_name)
            short_value = _format_param_value(param_value)
            suffix_parts.append(f"{short_name}{short_value}")
        
        # Create config and suffix
        config = TrainingConfig(**config_dict)
        suffix = "_".join(suffix_parts)
        configs.append((config, suffix))
    
    return configs


def _is_structural_list(param_name: str) -> bool:
    """Check if a parameter is a structural list (not meant for sweeping).
    
    Structural lists are single list values that define structure:
    - hidden_layers: [64, 64] defines layer sizes
    - feature_dims: [32, 32] defines feature dimensions  
    - video_resolution: [640, 480] defines width and height
    
    To sweep these, use nested lists:
    - hidden_layers: [[32, 32], [64, 64]] sweeps over architectures
    """
    structural = {'hidden_layers', 'feature_dims', 'video_resolution'}
    return param_name in structural


def _shorten_param_name(name: str) -> str:
    """Shorten parameter name for experiment naming.
    
    Examples:
        num_iters -> iters
        learning_rate -> lr
        batch_size -> bs
        num_epochs -> epochs
    """
    shortcuts = {
        'num_iters': 'iters',
        'learning_rate': 'lr',
        'batch_size': 'bs',
        'num_epochs': 'epochs',
        'num_envs': 'envs',
        'num_samples': 'samples',
        'noise_level': 'noise',
        'num_policy_samples': 'psamples',
        'num_knots': 'knots',
        'plan_horizon': 'horizon',
        'spline_type': 'spline',
        'exploration_noise_level': 'exnoise',
        'num_elites': 'elites',
        'sigma_start': 'sigma',
        'temperature': 'temp',
        'controller_type': 'ctrl',
        'architecture': 'arch',
        'hidden_layers': 'layers',
        'feature_dims': 'feats',
    }
    return shortcuts.get(name, name)


def _format_param_value(value: Any) -> str:
    """Format parameter value for experiment naming.
    
    Examples:
        0.001 -> 1e3
        0.01 -> 0.01
        10 -> 10
        "predictive_sampling" -> "ps"
        [64, 64] -> "64x64"
        [32, 32, 32] -> "32x32x32"
    """
    if isinstance(value, list):
        # Format list as "x"-separated string
        return "x".join(str(v) for v in value)
    elif isinstance(value, float):
        # Use scientific notation for small values
        if value < 0.01:
            return f"{value:.0e}".replace('e-0', 'e-').replace('e-', 'e')
        return str(value)
    elif isinstance(value, str):
        # Shorten common string values
        shortcuts = {
            'predictive_sampling': 'ps',
            'evosax': 'evo',
            'mppi': 'mppi',
            'cem': 'cem',
            'mlp': 'mlp',
            'cnn': 'cnn',
            'zero': 'zoh',
            'linear': 'lin',
            'cubic': 'cub',
        }
        return shortcuts.get(value, value)
    else:
        return str(value)


def print_sweep_summary(configs: list[tuple[TrainingConfig, str]]) -> None:
    """Print a summary of the hyperparameter sweep.
    
    Args:
        configs: List of (config, suffix) tuples from expand_config_sweep.
    """
    print(f"\n{'='*80}")
    print(f"Hyperparameter Sweep: {len(configs)} configurations")
    print(f"{'='*80}\n")
    
    # Find sweep parameters by comparing first and last config
    if len(configs) > 1:
        first_dict = configs[0][0].__dict__
        sweep_params = set()
        
        for config, _ in configs[1:]:
            for key, value in config.__dict__.items():
                if first_dict[key] != value:
                    sweep_params.add(key)
        
        print(f"Sweeping over: {', '.join(sorted(sweep_params))}\n")
    
    # Print each configuration
    for i, (config, suffix) in enumerate(configs, 1):
        exp_name = f"{config.task_name}_{suffix}" if suffix else config.task_name
        if config.experiment_name:
            exp_name = f"{config.experiment_name}_{suffix}" if suffix else config.experiment_name
        
        print(f"{i:2d}. {exp_name}")
    
    print(f"\n{'='*80}\n")
