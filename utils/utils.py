
import argparse
import os
import random
from typing import List, Union, Optional, Dict, Tuple
import gc

import numpy as np
import pandas as pd

import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

import wandb

from transformers import AutoProcessor, LlavaForConditionalGeneration, set_seed  # noqa: F401
import probing_utils as utils
import utils.utils



def set_wandb_mode(mode: Optional[str]):
    if mode is None:
        return
    if mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = mode
        os.environ.pop("WANDB_DISABLED", None)



def set_global_seed(seed: int):
    if seed is None:
        return
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import pandas as pd
from typing import Optional, Set, Tuple

import pandas as pd
from typing import Optional, Set, Dict

def load_csv_datasets(
    data_csv: str,
    *,
    train_csv: Optional[str] = None,
    required_cols: Optional[Set[str]] = None,
    n_samples: Optional[int] = None,
    train_n_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Load CSV files, validate required columns, and (optionally) sample rows.
    Always returns a dict so callers don't need to handle Optional unpacking.

    Parameters
    ----------
    data_csv : str
        Path to the evaluation/test CSV file.
    train_csv : Optional[str], default None
        (Optional) Path to the training CSV file. If omitted, the result
        will not contain a "train" key.
    required_cols : Optional[Set[str]], default {"image_path", "question", "label"}
        Set of required columns that must be present in the CSV files.
    n_samples : Optional[int], default None
        Number of samples to use from the evaluation/test dataset.
        If None, the entire dataset is used. Sampling only happens if n_samples < len(df).
    train_n_samples : Optional[int], default None
        Number of samples to use from the training dataset.
        If None, the entire dataset is used. Sampling only happens if train_n_samples < len(train_df).
    seed : int, default 42
        Random seed for reproducible sampling.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with at least {"data": df}. If train_csv is provided,
        the dictionary will also include {"train": train_df}.

    Raises
    ------
    ValueError
        If any required column is missing from a CSV.
    """
    if required_cols is None:
        required_cols = {"image_path", "question", "label"}

    # Load evaluation/test CSV
    df = pd.read_csv(data_csv)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in CSV: {missing}")

    # Optional sampling for evaluation/test dataset
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    result: Dict[str, pd.DataFrame] = {"data": df}

    # Optionally load training CSV
    if train_csv:
        train_df = pd.read_csv(train_csv)
        missing_train = required_cols - set(train_df.columns)
        if missing_train:
            raise ValueError(f"Missing required column(s) in train CSV: {missing_train}")

        if train_n_samples is not None and train_n_samples < len(train_df):
            train_df = train_df.sample(n=train_n_samples, random_state=seed).reset_index(drop=True)

        result["train"] = train_df

    return result


def safe_stack(feat_list: List[np.ndarray], dtype=np.float32) -> Optional[np.ndarray]:
    """Preallocate and copy row-by-row to avoid transient double-allocation."""
    if not feat_list:
        return None
    first = np.asarray(feat_list[0])
    out = np.empty((len(feat_list),) + first.shape, dtype=dtype)
    for i, a in enumerate(feat_list):
        ai = np.asarray(a)
        if ai.shape != first.shape:
            raise ValueError(f"Inconsistent feature shape at {i}: {ai.shape} vs {first.shape}")
        out[i] = ai.astype(dtype, copy=False)
    return out

def parse_subset_layers(arg_val: Optional[str], num_layers: int) -> List[int]:
    if not arg_val:
        return list(range(num_layers))
    raw = [s.strip() for s in arg_val.split(",") if s.strip() != ""]
    idxs: List[int] = []
    for s in raw:
        i = int(s)
        if i < 0:
            i = num_layers + i  # negative index from end
        if not (0 <= i < num_layers):
            raise ValueError(f"subset_layers index out of range after normalization: {i} (num_layers={num_layers})")
        idxs.append(i)
    # keep order but uniquify
    seen = set()
    ordered = []
    for i in idxs:
        if i not in seen:
            ordered.append(i)
            seen.add(i)
    return ordered