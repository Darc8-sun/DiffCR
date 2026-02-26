import importlib
from typing import Any, Mapping

import numpy as np
from torch import nn


def get_obj_from_str(string: str, reload: bool = False) -> object:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def frozen_module(module: nn.Module) -> None:
    module.eval()
    module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False


def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool = False,
                    load_compressor: bool = False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)

    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")

    # Handle Compression Model
    compress_model = {}
    for key, value in state_dict.items():
        if 'compress_model' in key:
            compress_model[key[len('compress_model.'):]] = value
    for key, value in compress_model.items():
        state_dict.pop(f'compress_model.{key}')

    if (
            is_model_key_starts_with_module and
            (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items() if 'compress_model' not in key}
    if (
            (not is_model_key_starts_with_module) and
            is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items() if 'compress_model' not in key}

    print(f'LOAD COMPRESSOR: {load_compressor}')
    if load_compressor:
        # from elic.elic_centropy import ELIC_E1 as compressor
        # from compressai.models import Compressor as compressor
        # from elic import ELIC as compressor
        # from main_elic import GetELICcompressor
        # from omegaconf import OmegaConf
        # configs=OmegaConf.load("configs/elic_latent_eval.yaml")
        # model.compress_model =GetELICcompressor(configs)
        model.compress_model.load_state_dict(compress_model)
        #
        # model.compress_model = compressor.from_state_dict(compress_model)
        # model.compress_model.update(force=True)
    model.load_state_dict(state_dict, strict=strict)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def resume_from_checkpoint(model: nn.Module, state_dict: Mapping[str, Any], strict: bool = False,
                           load_compressor: bool = False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)

    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")

    # Handle Compression Model
    compress_model = {}
    for key, value in state_dict.items():
        if 'compress_model' in key:
            compress_model[key[len('compress_model.'):]] = value
    for key, value in compress_model.items():
        state_dict.pop(f'compress_model.{key}')

    if (
            is_model_key_starts_with_module and
            (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items() if 'compress_model' not in key}
    if (
            (not is_model_key_starts_with_module) and
            is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items() if 'compress_model' not in key}

    print(f'LOAD COMPRESSOR: {load_compressor}')
    if load_compressor:
        # from elic.elic_centropy import ELIC_E1 as compressor
        # from compressai.models import Compressor as compressor
        # from elic import ELIC as compressor
        # from main_elic import GetELICcompressor
        # from omegaconf import OmegaConf
        # configs=OmegaConf.load("configs/elic_latent_eval.yaml")
        # model.compress_model =GetELICcompressor(configs)
        model.compress_model.load_state_dict(compress_model)
        #
        # model.compress_model = compressor.from_state_dict(compress_model)
        # model.compress_model.update(force=True)
    model.load_state_dict(state_dict, strict=strict)