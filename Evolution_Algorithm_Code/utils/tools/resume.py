""" Model creation / weight loading / state_dict helpers

Patch goals:
 - Robustly normalize checkpoint keys (module./model./backbone./net.)
 - Detect checkpoint head size(s) and ONLY drop classifier params on a true mismatch
 - Re-init head if (and only if) we dropped it
 - Concise + useful logging of missing keys
 - Python 3.8/3.9 typing compatible
"""
import logging
import os
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

# --------- helpers ---------

_PREFIXES = ("module.", "model.", "backbone.", "net.")

def _normalize_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip any chained prefixes like 'module.model.backbone.'."""
    out = OrderedDict()
    for k, v in state_dict.items():
        name = k
        changed = True
        while changed:
            changed = False
            for p in _PREFIXES:
                if name.startswith(p):
                    name = name[len(p):]
                    changed = True
        out[name] = v
    return out

def _expected_out_features(model: nn.Module) -> Optional[int]:
    """Infer classifier output dim from common heads (fc/classifier/head/linear)."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.out_features
    for attr in ("classifier", "head", "linear"):
        mod = getattr(model, attr, None)
        if mod is not None:
            last = None
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    last = m
            if last is not None:
                return last.out_features
    return None

def _find_ckpt_heads(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, int]]:
    """Return list of (prefix, out_features) heads found in ckpt."""
    heads: List[Tuple[str, int]] = []
    for pref in ("fc.", "classifier.", "head.", "linear."):
        w = pref + "weight"
        if w in state_dict and state_dict[w].dim() == 2:
            heads.append((pref, int(state_dict[w].shape[0])))
    return heads

def _drop_head_if_mismatch(state_dict: Dict[str, torch.Tensor], model: nn.Module, log: bool) -> bool:
    """Only drop head weights if the checkpoint head size != model expected size."""
    expected = _expected_out_features(model)
    heads = _find_ckpt_heads(state_dict)

    if log:
        if heads:
            _logger.info(
                "Checkpoint head(s): " + ", ".join([f"{p}->{n}" for p, n in heads]) +
                (f" | model expects {expected}" if expected is not None else "")
            )
        else:
            _logger.info("No explicit head found in checkpoint (fc./classifier./head./linear.).")

    if expected is None or not heads:
        return False

    dropped = False
    for prefix, out_features in heads:
        if out_features != expected:
            state_dict.pop(prefix + "weight", None)
            state_dict.pop(prefix + "bias", None)
            dropped = True

    if dropped and log:
        _logger.info("Dropped mismatched classifier params; kept matching ones.")
    return dropped

def _reinit_head(model: nn.Module, log: bool):
    """Re-initialize classifier head(s)."""
    did = False
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        nn.init.normal_(model.fc.weight, std=0.01)
        if model.fc.bias is not None:
            nn.init.zeros_(model.fc.bias)
        did = True
    for attr in ("classifier", "head", "linear"):
        mod = getattr(model, attr, None)
        if mod is not None:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    did = True
    if did and log:
        _logger.info("Re-initialized classifier head for current num_classes.")

def _log_missing(missing, unexpected, log_info: bool):
    if not log_info:
        return
    SUPPRESS = ("fc.", "classifier.", "head.", "linear.")
    sample_n = 12
    unknown_missing = [k for k in missing if not k.startswith(SUPPRESS)]
    if unknown_missing:
        _logger.info(
            f"Missing keys (sample {min(sample_n, len(unknown_missing))}/{len(missing)}): "
            f"{unknown_missing[:sample_n]}"
        )
    else:
        _logger.info(f"Missing keys: {len(missing)} (classifier/head adaptation or benign differences)")
    if unexpected:
        _logger.info(f"Unexpected keys: {len(unexpected)}")

# --------- public API ---------

def load_state_dict(checkpoint_path: str, log_info: bool = True, use_ema: bool = False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict_key = ""
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get("state_dict_ema", None) is not None:
                state_dict_key = "state_dict_ema"
            elif use_ema and checkpoint.get("model_ema", None) is not None:
                state_dict_key = "model_ema"
            elif "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            elif "model" in checkpoint:
                state_dict_key = "model"

        if state_dict_key:
            state_dict = _normalize_keys(checkpoint[state_dict_key])
        else:
            state_dict = _normalize_keys(checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if log_info:
            _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key or "raw", checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    log_info: bool = True, use_ema: bool = False, strict: bool = True):
    state_dict = load_state_dict(checkpoint_path, log_info, use_ema)
    dropped = _drop_head_if_mismatch(state_dict, model, log=log_info)

    # If we dropped head weights, allow missing head keys; otherwise be strict.
    strict_flag = False if dropped else bool(strict)
    incompatible = model.load_state_dict(state_dict, strict=strict_flag)
    # PyTorch returns IncompatibleKeys(missing_keys, unexpected_keys)
    missing = getattr(incompatible, "missing_keys", []) if hasattr(incompatible, "__iter__") else incompatible[0]
    unexpected = getattr(incompatible, "unexpected_keys", []) if hasattr(incompatible, "__iter__") else incompatible[1]

    if dropped or any(k.startswith(("fc.", "classifier.", "head.", "linear.")) for k in missing):
        _reinit_head(model, log=log_info)

    if log_info:
        _logger.info("Loaded  from checkpoint '{}'".format(checkpoint_path))
        _log_missing(missing, unexpected, log_info)

def resume_checkpoint(model: nn.Module, checkpoint_path: str,
                      optimizer=None, loss_scaler=None, log_info: bool = True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and ("state_dict" in checkpoint or "model" in checkpoint):
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            if "state_dict" in checkpoint:
                checkpoint_model_dict = _normalize_keys(checkpoint["state_dict"])
            else:
                checkpoint_model_dict = _normalize_keys(checkpoint["model"])

            dropped = _drop_head_if_mismatch(checkpoint_model_dict, model, log=log_info)
            incompatible = model.load_state_dict(checkpoint_model_dict, strict=not dropped)
            missing = getattr(incompatible, "missing_keys", []) if hasattr(incompatible, "__iter__") else incompatible[0]
            unexpected = getattr(incompatible, "unexpected_keys", []) if hasattr(incompatible, "__iter__") else incompatible[1]

            if dropped or any(k.startswith(("fc.", "classifier.", "head.", "linear.")) for k in missing):
                _reinit_head(model, log=log_info)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    _logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and getattr(loss_scaler, "state_dict_key", None) in checkpoint:
                if log_info:
                    _logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]
                if "version" in checkpoint and checkpoint["version"] > 1:
                    resume_epoch += 1

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint.get("epoch", "?")))
                _log_missing(missing, unexpected, log_info)
        else:
            tensor_dict = _normalize_keys(checkpoint) if isinstance(checkpoint, dict) else checkpoint
            dropped = _drop_head_if_mismatch(tensor_dict, model, log=log_info)
            incompatible = model.load_state_dict(tensor_dict, strict=not dropped)
            missing = getattr(incompatible, "missing_keys", []) if hasattr(incompatible, "__iter__") else incompatible[0]
            unexpected = getattr(incompatible, "unexpected_keys", []) if hasattr(incompatible, "__iter__") else incompatible[1]
            if dropped or any(k.startswith(("fc.", "classifier.", "head.", "linear.")) for k in missing):
                _reinit_head(model, log=log_info)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
                _log_missing(missing, unexpected, log_info)
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
