""" Add formulae intensities """
import json
import numpy as np
import argparse
import copy
from pathlib import Path

import ms_pred.magma.run_magma as run_magma
import ms_pred.common as common


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--pred-form-folder", action="store")
    parser.add_argument("--true-form-folder", action="store")
    parser.add_argument("--out-form-folder", action="store", default=None)
    parser.add_argument("--binned-add", action="store_true", default=False)
    parser.add_argument("--add-raw", action="store_true", default=False)
    return parser.parse_args()


def relabel_tree(
    pred_form_file: Path,
    true_form_file: Path,
    out_form_file: Path,
    add_binned: bool,
    add_raw,
):
    """relabel_tree.

    Args:
        pred_form_file (Path): pred_form_file
        true_form_file (Path): true_form_file
        out_form_file (Path): out_form_file
        add_binned (bool): add_binned
        add_raw (bool): add_raw

    Returns:
        None:
    """
    if not true_form_file.exists():
        return

    pred_form = json.load(open(pred_form_file, "r"))
    true_form = json.load(open(true_form_file, "r"))

    if true_form["output_tbl"] is None:
        return

    pred_tbl, true_tbl = pred_form.get("output_tbl", None), true_form["output_tbl"]
    if pred_tbl is None:
        pred_tbl = {"formula_mass_no_adduct": []}

    # Use rel inten
    true_form_to_inten = dict(zip(true_tbl["formula"], true_tbl["rel_inten"]))

    if add_binned:
        bins = np.linspace(0, 1500, 15000)
        true_pos = np.digitize(true_tbl["formula_mass_no_adduct"], bins)
        pred_pos = np.digitize(pred_tbl["formula_mass_no_adduct"], bins)
        bin_to_inten = dict()
        for i, j in zip(true_pos, true_tbl["rel_inten"]):
            bin_to_inten[i] = max(j, bin_to_inten.get(i, 0))
        new_intens = [bin_to_inten.get(i, 0) for i in pred_pos]
    else:
        new_intens = [true_form_to_inten.get(i, 0.0) for i in pred_tbl["formula"]]

    if add_raw:
        raw_spec = list(zip(true_tbl["formula_mass_no_adduct"], true_tbl["rel_inten"]))
        pred_tbl["raw_spec"] = raw_spec

    pred_tbl["rel_inten"] = new_intens
    pred_tbl["ms2_inten"] = new_intens
    with open(out_form_file, "w") as fp:
        json.dump(pred_form, fp, indent=2)


def main():
    """main."""
    args = get_args()
    pred_form_folder = Path(args.pred_form_folder)
    true_form_folder = Path(args.true_form_folder)
    out_form_folder = args.out_form_folder
    add_binned = args.binned_add
    add_raw = args.add_raw

    if out_form_folder is None:
        out_form_folder = pred_form_folder
    out_form_folder = Path(out_form_folder)
    out_form_folder.mkdir(exist_ok=True)

    num_workers = args.num_workers
    pred_form_files = list(pred_form_folder.glob("*.json"))
    true_form_files = [
        true_form_folder / i.name.replace("pred_", "") for i in pred_form_files
    ]
    out_form_files = [out_form_folder / i.name for i in pred_form_files]
    arg_dicts = [
        {
            "pred_form_file": i,
            "true_form_file": j,
            "out_form_file": k,
            "add_raw": add_raw,
            "add_binned": add_binned,
        }
        for i, j, k in zip(pred_form_files, true_form_files, out_form_files)
    ]

    # Run
    wrapper_fn = lambda arg_dict: relabel_tree(**arg_dict)
    # Debug
    if num_workers == 0:
        [wrapper_fn(i) for i in arg_dicts]
    else:
        common.chunked_parallel(arg_dicts, wrapper_fn, max_cpu=num_workers)


if __name__ == "__main__":
    main()
