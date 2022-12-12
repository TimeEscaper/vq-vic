import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn

from nip import nip
from typing import List, Union, Iterable, Dict
from torch.nn.modules.batchnorm import _NormBase

_LOGGER = logging.getLogger(__name__)

OutputDict = Dict[str, torch.Tensor]
ContextType = Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]


def flatten_modules(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]) -> List[nn.Module]:
    """Flattens module into iterable over modules.

    Notes
    -----
    Flattens a module or an iterable of modules into a list of its leaf modules
    (modules with no children) and parent modules that have parameters directly themselves.

    Parameters
    ----------
    modules:
        A given module or an iterable of modules.

    Returns
    -------
    List of modules
    """
    if isinstance(modules, nn.ModuleDict):
        modules = modules.values()

    if isinstance(modules, Iterable):
        _modules = []
        for m in modules:
            _modules.extend(flatten_modules(m))

    else:
        _modules = modules.modules()

    # Capture all leaf modules as well as parent modules that have parameters directly themselves
    return [m for m in _modules if not list(m.children()) or m._parameters]


def make_trainable(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]) -> None:
    """Unfreezes the parameters of the provided modules.

    Parameters
    ----------
    modules:
        A given module or an iterable of modules
    """
    modules = flatten_modules(modules)
    for module in modules:
        # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
        for param in module.parameters(recurse=False):
            param.requires_grad = True


def freeze(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]],
           train_norm: bool = True) -> None:
    """Freeze module parameters for inference."""
    modules = flatten_modules(modules)
    for mod in modules:
        if isinstance(mod, _NormBase) and train_norm:
            make_trainable(mod)
        else:
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in mod.parameters(recurse=False):
                param.requires_grad = False


def unfreeze(modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]],) -> None:
    """Unfreeze module parameters for training."""
    make_trainable(modules)


@nip
def pretrained(model: nn.Module,
               ckpt_path: str,
               eval_mode: bool = True,
               freeze_model: bool = True,
               module_name: str = None):
    """Loads and freezes pretrained model.

    Parameters
    ----------
    model:
        torch model.
    ckpt_path:
        path to checkpoint.
    eval_mode:
        whether to set the model into eval mode.
    freeze_model:
        whether to freeze the model.
    module_name:
        if specified will load this name from 'state_dict' of Pytorch Lightning checkpoint.

    Returns
    -------
    loaded model.
    """
    if hasattr(model, 'load_from_checkpoint'):
        model.load_from_checkpoint(ckpt_path)
    else:
        if module_name:
            module_name += '.'
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict_to_load = {}
            for key, value in state_dict.items():
                if key.startswith(module_name):
                    state_dict_to_load[key.removeprefix(module_name)] = value
            state_dict = state_dict_to_load
        else:
            state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
    _LOGGER.info(f"Loaded pretrained weights from {ckpt_path}")
    if eval_mode:
        model.eval()
    if freeze_model:
        freeze(model)
    return model
