from pathlib import Path
import pandas as pd
import argparse
import pickle
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle-file", default="data/retrieval/pubchem/pubchem_formula_map.p"
    )
    parser.add_argument("--dataset-labels")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_labels = args.dataset_labels
    pickle_file = args.pickle_file

    dataset_name = Path(data_labels).parent.stem
    pubchem_sub_out = f"data/retrieval/pubchem/pubchem_formula_map_{dataset_name}.p"

    full_map = pickle.load(open(pickle_file, "rb"))
    uniq_forms = pd.unique(pd.read_csv(data_labels, sep="\t")["formula"])

    sub_map = {uniq_form: full_map.get(uniq_form, []) for uniq_form in uniq_forms}

    with open(pubchem_sub_out, "wb") as f:
        pickle.dump(sub_map, f)
