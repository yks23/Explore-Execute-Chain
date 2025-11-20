# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn

def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    print(f"Loading custom reward function from: {file_path}")
    if not file_path:
        return None
  
    # Try to find the file using multiple strategies for robustness
    resolved_path = None
    search_paths = []
    
    # Strategy 1: Use path as-is (absolute or relative to CWD)
    search_paths.append(file_path)
    
    # Strategy 2: Relative to current working directory
    cwd = os.getcwd()
    search_paths.append(os.path.join(cwd, file_path))
    
    # Strategy 3: Relative to script directory (verl/verl/trainer/ppo/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(script_dir, file_path))
    
    # Strategy 4: Relative to verl root (go up from script_dir)
    verl_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    search_paths.append(os.path.join(verl_root, file_path))
    
    # Strategy 5: Relative to project root (go up one more level)
    project_root = os.path.abspath(os.path.join(verl_root, ".."))
    search_paths.append(os.path.join(project_root, file_path))
    
    # Strategy 6: If path contains 'verl/', try to find verl directory
    if 'verl/' in file_path or 'verl\\' in file_path:
        # Extract the part after 'verl/'
        parts = file_path.replace('\\', '/').split('verl/')
        if len(parts) > 1:
            relative_to_verl = parts[-1]
            search_paths.append(os.path.join(verl_root, relative_to_verl))
    
    # Try each path
    for path in search_paths:
        if os.path.exists(path):
            resolved_path = path
            print(f"✓ Found reward function file at: {resolved_path}")
            break
    
    if resolved_path is None:
        error_msg = f"Reward function file not found. Tried:\n"
        error_msg += f"  Original path: {file_path}\n"
        error_msg += f"  Current directory: {cwd}\n"
        error_msg += "\nSearched locations:\n"
        for i, path in enumerate(search_paths, 1):
            error_msg += f"  {i}. {path}\n"
        raise FileNotFoundError(error_msg)

    spec = importlib.util.spec_from_file_location("custom_module", resolved_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{resolved_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    assert function_name is not None
    if not hasattr(module, function_name):
        available_functions = [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name))]
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{resolved_path}'.\n"
            f"Available functions: {', '.join(available_functions)}"
        )

    print(f"✓ Using customized reward function '{function_name}' from '{resolved_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    return partial(_call_with_kwargs, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)

    if compute_score is not None:
        # Use custom reward function if provided
        final_compute_score = compute_score
    else:
        # Use E2C reward function (refactored from inline implementation)
        from verl.utils.reward_score import e2c
        
        # Create a partial function with the config parameters
        use_constrain_reward = config.reward_model.get("use_constrain_reward", False)
        final_compute_score = partial(
            e2c.compute_score,
            use_constrain_reward=use_constrain_reward
        )
    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)
