""" make_splits.py

Make train-test-val splits by compound uniqueness.


"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data-dir", default='data/spec_datasets/gnps2015_debug',
                        help="Data directory")
    args.add_argument("--label-file",
                      default='data/spec_datasets/gnps2015_debug/labels.tsv',
                        help="Path to label file.")
    args.add_argument("--ionization", type=str, default=None,
                      help="Ion that the user want to focused on (if applicable)")
    args.add_argument("--val-frac", type=float, default=0.1,
                        help="Percentage of validation data out of all training data.")
    args.add_argument("--test-frac", type=float, default=0.1,
                        help="Percentage of test data out of all data.")
    args.add_argument("--seed", type=int, default=22,
                        help="Random seed be reproducible")
    args.add_argument("--split-name", default=None,
                        help="split name prefix")
    return args.parse_args()


def make_splits(args):
    label_path = Path(args.label_file)
    df = pd.read_csv(label_path, sep="\t")
    split_suffix = args.split_name
    seed = args.seed

    if split_suffix is None:
        split_suffix = f"split_{seed}.tsv"

    if args.ionization is not None:
        df = df[df['ionization'] == args.ionization]

    ikey_set = set(df["inchikey"].values)

    train_frac, _test_frac = (1-args.test_frac), args.test_frac
    num_train = int(train_frac * len(ikey_set))

    # Divide by compounds
    full_formula_list = list(ikey_set)
    np.random.seed(seed)
    np.random.shuffle(full_formula_list)
    train = set(full_formula_list[:num_train])
    test = set(full_formula_list[num_train:])
 
    output_dir = Path(args.data_dir) / 'splits'
    if not output_dir.is_dir():
        output_dir.mkdir(exist_ok=True)

    val_fraction = args.val_frac
    fold_num = 0
    fold_name = f"Fold_{fold_num}"
    val_num = int(len(train)*val_fraction)
    np.random.seed(seed)
    val = set(np.random.choice(list(train), val_num, replace=False))

    # Remove val formulae inds from train formulae
    train = train.difference(val)

    print(f"Num train total formulae: {len(train)}")
    print(f"Num val total formulae: {len(val)}")
    print(f"Num test total formulae: {len(test)}")
    
    split_data = {'spec': [], fold_name: []}
    for _, row in df.iterrows():
        spec_form = row["inchikey"]
        spec_fn = row['spec']

        if spec_form in train:
            fold = "train"
        elif spec_form in test:
            fold = "test"
        elif spec_form in val:
            fold = "val"
        else:
            fold = "exclude"
        split_data['spec'].append(spec_fn)
        split_data[fold_name].append(fold)
        
    assert len(split_data['spec']) == df.shape[0]
    assert len(split_data[fold_name]) == df.shape[0]
    export_df = pd.DataFrame(split_data)
    export_df = export_df.sort_values('spec', ascending=True)
    export_df.to_csv(
        output_dir / split_suffix , sep='\t', index=False
    )


if __name__=="__main__":
    args = get_args()
    make_splits(args)
