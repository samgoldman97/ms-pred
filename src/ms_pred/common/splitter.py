""" splitter.py """

import logging
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd


def get_splits(
    names: List[str],
    split_file: str,
    val_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get_splits.
    Args:
        names (List[str]): Names to be split
        split_file (str): Split file
        val_frac (float): Fraction of validation
    Return:
        Train, val, test indices
    """

    if not Path(split_file).exists():
        logging.info(f"Unable to find {split_file}")
        raise ValueError()

    # Resetting num folds to 10 regardless
    split_df = pd.read_csv(split_file, sep="\t")

    folds = set(split_df.columns)
    folds.remove("spec")
    num_folds = len(folds)
    folds = sorted(list(folds))
    if len(folds) == 0:
        raise ValueError
    elif len(folds) > 1:
        logging.info(f"Found {num_folds} folds; choosing one")
        fold = folds[0]
    else:
        fold = folds[0]

    fold_entries = split_df[fold]
    names_to_index = dict(zip(names, np.arange(len(names))))
    train_entries = fold_entries == "train"
    val_entries = fold_entries == "val"
    test_entries = fold_entries == "test"
    train_inds = [
        names_to_index.get(i)
        for i in split_df["spec"][train_entries]
        if i in names_to_index
    ]
    test_inds = np.array(
        [
            names_to_index.get(i)
            for i in split_df["spec"][test_entries]
            if i in names_to_index
        ]
    )
    val_inds = np.array(
        [
            names_to_index.get(i)
            for i in split_df["spec"][val_entries]
            if i in names_to_index
        ]
    )

    convert = lambda x: np.array(list(x))
    return convert(train_inds), convert(val_inds), convert(test_inds)


def random_split(names: List[str], split_sizes=(0.8, 0.1, 0.1)):
    """Randomly split indices into proportions defined"""

    train_size, val_size, test_size = split_sizes
    dataset_size = len(names)
    first_ind = int(np.ceil(dataset_size * train_size))
    second_ind = first_ind + int(np.ceil(dataset_size * val_size))
    third_ind = second_ind + int(np.ceil(dataset_size * test_size))

    all_inds = np.arange(dataset_size)
    np.random.shuffle(all_inds)

    train_smis = all_inds[:first_ind]
    val_smis = all_inds[first_ind:second_ind]
    test_smis = all_inds[second_ind:third_ind]
    return list(train_smis), list(val_smis), list(test_smis)
