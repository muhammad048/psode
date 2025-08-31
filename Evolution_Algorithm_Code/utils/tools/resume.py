""" Model creation / weight loading / state_dict helpers

Based on Ross Wightman utils; patched to:
 - drop mismatched classifier heads safely
 - re-init head params when adapting checkpoints
 - be Python 3.8/3.9 typing compatible
"""
import logging
import os
from collections import OrderedDict
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        out[name] = v
    return out

def _expected_out_features(model: nn.Module) -> Optional[int]:
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model.fc.out_features
    for attr in ('classifier', 'head'):
        mod = getattr(model, attr, None)
        if mod is not None:
            last = None
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if last is not None:
                return last.out_features
    return None

def _drop_head_if_mismatch(state_dict: Dict[str, torch.Tensor], model: nn.Module, log: bool) -> bool:
    expected = _expected_out_features(model)
    if expected is None:
        return False
    dropped = False
    for prefix in ('fc.', 'classifier.', 'head.'):
        w_key, b_key = prefix + 'weight', prefix + 'bias'
        if w_key in state_dict:
            if state_dict[w_key].shape[0] != expected:
                state_dict.pop(w_key, None)
                state_dict.pop(b_key, None)
                dropped = True
    if dropped and log:
        _logger.info(
            "Dropped classifier params from checkpoint due to class-count mismatch "
            f"(expected out_features={expected})."
        )
    return dropped

def _reinit_head(model: nn.Module, log: bool):
    did = False
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        nn.init.normal_(model.fc.weight, std=0.01)
        if model.fc.bias is not None:
            nn.init.zeros_(model.fc.bias)
        did = True
    for attr in ('classifier', 'head'):
        mod = getattr(model, attr, None)
        if mod is not None:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    did = True
    if did and log:
        _logger.info("Re-initialized classifier head parameters for the current num_classes.")

def load_state_dict(checkpoint_path, log_info: bool = True, use_ema: bool = False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            state_dict = _strip_module_prefix(state_dict)
        else:
            state_dict = _strip_module_prefix(checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if log_info:
            _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    log_info: bool = True, use_ema: bool = False, strict: bool = True):
    state_dict = load_state_dict(checkpoint_path, log_info, use_ema)
    dropped = _drop_head_if_mismatch(state_dict, model, log=log_info)
    strict = False if dropped else bool(strict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if dropped or any(k.startswith(('fc.', 'classifier.', 'head.')) for k in missing):
        _reinit_head(model, log=log_info)
    if log_info:
        _logger.info("Loaded  from checkpoint '{}'".format(checkpoint_path))
        if missing:
            _logger.info("Missing keys: {}".format(len(missing)))
        if unexpected:
            _logger.info("Unexpected keys: {}".format(len(unexpected)))

def resume_checkpoint(model: nn.Module, checkpoint_path: str,
                      optimizer=None, loss_scaler=None, log_info: bool = True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model' in checkpoint):
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            if 'state_dict' in checkpoint:
                checkpoint_model_dict = _strip_module_prefix(checkpoint['state_dict'])
            else:
                checkpoint_model_dict = _strip_module_prefix(checkpoint['model'])
            _drop_head_if_mismatch(checkpoint_model_dict, model, log=log_info)
            missing, unexpected = model.load_state_dict(checkpoint_model_dict, strict=False)
            if any(k.startswith(('fc.', 'classifier.', 'head.')) for k in missing):
                _reinit_head(model, log=log_info)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and getattr(loss_scaler, 'state_dict_key', None) in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint.get('epoch', '?')))
        else:
            tensor_dict = _strip_module_prefix(checkpoint) if isinstance(checkpoint, dict) else checkpoint
            _drop_head_if_mismatch(tensor_dict, model, log=log_info)
            missing, unexpected = model.load_state_dict(tensor_dict, strict=False)
            if any(k.startswith(('fc.', 'classifier.', 'head.')) for k in missing):
                _reinit_head(model, log=log_info)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
