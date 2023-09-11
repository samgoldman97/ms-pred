""" make_splits.py

Make train-test-val splits by compound uniqueness.


"""
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ms_pred import common


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data-dir", default="data/spec_datasets/gnps2015_debug", help="Data directory"
    )
    args.add_argument(
        "--label-file",
        default="data/spec_datasets/gnps2015_debug/labels.tsv",
        help="Path to label file.",
    )
    args.add_argument(
        "--ionization",
        type=str,
        default=None,
        help="Ion that the user want to focused on (if applicable)",
    )
    args.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Percentage of validation data out of all training data.",
    )
    args.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Percentage of validation data out of all training data.",
    )
    args.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Percentage of test data out of all data.",
    )
    args.add_argument(
        "--seed", type=int, default=22, help="Random seed be reproducible"
    )
    args.add_argument("--split-name", default=None, help="split name prefix")
    args.add_argument(
        "--split-type",
        default="inchikey",
        help="Type of split; normally split on structures",
        choices=["inchikey", "scaffold", "fingerprint"],
    )
    args.add_argument("--greedy-pack", default=False, action="store_true")
    return args.parse_args()


def get_scaffold(smiles):
    """
    Given a SMILES string, returns the scaffold of the molecule using the Murcko framework.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
        return Chem.MolToInchiKey(scaffold)
    except:
        print(f"Failed on {smiles}, return none")
        return ""


def make_splits(args):
    label_path = Path(args.label_file)
    df = pd.read_csv(label_path, sep="\t")
    split_suffix = args.split_name
    seed = args.seed
    split_type = args.split_type
    greedy_pack = args.greedy_pack

    if split_suffix is None:
        split_suffix = f"split_{seed}.tsv"

    if args.ionization is not None:
        df = df[df["ionization"] == args.ionization]

    if split_type == "inchikey":
        obj_set = set(df["inchikey"].values)
        spec_to_obj = dict(df[["spec", "inchikey"]].values)
    elif split_type == "scaffold":
        smis = df["smiles"].values
        specs = df["spec"].values
        scaffolds = common.chunked_parallel(smis, get_scaffold)

        obj_set = set(scaffolds)
        spec_to_obj = dict(zip(specs, scaffolds))
    elif split_type == "fingerprint":

        get_fp = lambda x: common.get_morgan_fp_smi(x, nbits=512)

        # debug_num = 500
        smis = df["smiles"].values
        specs = df["spec"].values
        fps = common.chunked_parallel(smis, get_fp)
        fps = np.vstack(fps)

        print("Computing tanimoto")
        intersect = np.einsum("ij, kj -> ik", fps, fps)
        union = fps.sum(-1)[None, :] + fps.sum(-1)[:, None] - intersect
        tani_dist = 1 - intersect / (union + 1e-22)
        print("Done with tanimoto")

        print("Running agglom clustering")
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=20,
            linkage="complete",
            affinity="precomputed",
        )
        # affinity=tani_dist,)
        print("Done agglom clustering")
        clusts = model.fit_predict(tani_dist)
        obj_set = set(clusts)
        spec_to_obj = dict(zip(specs, clusts))

    else:
        raise NotImplementedError()

    # Compute nums in terms of specs
    train_frac, val_frac, test_frac = args.train_frac, args.val_frac, args.test_frac
    num_train = int(train_frac * len(spec_to_obj))
    num_val = int(args.val_frac * len(spec_to_obj))
    num_test = min(
        len(spec_to_obj) - num_train - num_val, int(test_frac * len(spec_to_obj))
    )

    if greedy_pack:
        # Pack low items
        # Get counts associated with each
        obj_to_counts = Counter(spec_to_obj.values())

        # Small to large sorted
        sorted_obj_set = sorted(list(obj_set), key=lambda x: obj_to_counts[x])
        get_num_specs = lambda x: np.sum([obj_to_counts[i] for i in x])

        train, val, test = set(), set(), set()
        for obj in sorted_obj_set:
            if get_num_specs(test) < num_test:
                test.add(obj)
            elif get_num_specs(val) < num_val:
                val.add(obj)
            else:
                train.add(obj)
    else:
        num_train = int((train_frac + val_frac) * len(obj_set))
        num_test = len(obj_set) - num_train - num_val

        # Divide by compounds
        full_obj_list = list(obj_set)
        np.random.seed(seed)
        np.random.shuffle(full_obj_list)
        train = set(full_obj_list[:num_train])
        test = set(full_obj_list[num_train:])

        val_fraction = args.val_frac
        val_num = int(len(train) * val_fraction)
        np.random.seed(seed)
        val = set(np.random.choice(list(train), val_num, replace=False))

        # Remove val formulae inds from train formulae
        train = train.difference(val)

    fold_num = 0
    fold_name = f"Fold_{fold_num}"
    output_dir = Path(args.data_dir) / "splits"
    if not output_dir.is_dir():
        output_dir.mkdir(exist_ok=True)

    print(f"Num train total formulae: {len(train)}")
    print(f"Num val total formulae: {len(val)}")
    print(f"Num test total formulae: {len(test)}")

    split_data = {"spec": [], fold_name: []}
    for _, row in df.iterrows():
        spec_fn = row["spec"]
        spec_obj = spec_to_obj[spec_fn]

        if spec_obj in train:
            fold = "train"
        elif spec_obj in test:
            fold = "test"
        elif spec_obj in val:
            fold = "val"
        else:
            fold = "exclude"
        split_data["spec"].append(spec_fn)
        split_data[fold_name].append(fold)

    assert len(split_data["spec"]) == df.shape[0]
    assert len(split_data[fold_name]) == df.shape[0]

    export_df = pd.DataFrame(split_data)
    export_df = export_df.sort_values("spec", ascending=True)
    export_df.to_csv(output_dir / split_suffix, sep="\t", index=False)


if __name__ == "__main__":
    args = get_args()
    make_splits(args)
