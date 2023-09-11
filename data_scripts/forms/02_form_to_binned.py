""" form_to_binned.py

Convert dag folder into a binned spec file

"""
import json
import argparse
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import ms_pred.common as common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--form-folder", action="store")
    parser.add_argument(
        "--min-inten",
        type=float,
        default=1e-5,
        help="Minimum intensity to call a peak in prediction",
    )
    parser.add_argument(
        "--max-peaks", type=int, default=100, help="Max num peaks to call"
    )
    parser.add_argument("--num-bins", type=int, default=1500, help="Num bins")
    parser.add_argument("--upper-limit", type=int, default=1500, help="upper lim")
    parser.add_argument("--out", action="store")
    return parser.parse_args()


def bin_forms(
    forms: dict, max_peaks: int, upper_limit: int, num_bins: int, min_inten: float
) -> np.ndarray:
    """bin_dag.

    Args:
        forms (dict): forms
        max_peaks (int): max_peaks
        upper_limit (int): upper_limit
        num_bins (int): num_bins
        min_inten (float): min_inten

    Returns:
        np.ndarray:
    """
    smiles = forms.get("smiles")

    tbl = forms.get("output_tbl")
    if tbl is None:
        tbl = {"rel_inten": [], "formula_mass_no_adduct": []}

    intens = tbl["rel_inten"]
    mz = tbl["formula_mass_no_adduct"]

    # Cap at min
    mz, intens = np.array(mz), np.array(intens)
    min_inten_mask = intens > min_inten
    mz = mz[min_inten_mask]
    intens = intens[min_inten_mask]

    # Keep 100 max
    argsort_intens = np.argsort(intens)[::-1][:max_peaks]
    mz = mz[argsort_intens]
    intens = intens[argsort_intens]
    spec_ar = np.vstack([mz, intens]).transpose(1, 0)

    # Bin intensities
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit, pool_fn="max")
    return binned, smiles


def bin_form_file(
    form_file: dict, max_peaks: int, upper_limit: int, num_bins: int, min_inten: float
):
    """bin_dag_file.

    Args:
        dag_file (dict): dag_file
        max_peaks (int): max_peaks
        upper_limit (int): upper_limit
        num_bins (int): num_bins
        min_inten (float): min_inten

    Returns:
        np.ndarray:
        str
    """
    forms = json.load(open(form_file, "r"))
    return bin_forms(forms, max_peaks, upper_limit, num_bins, min_inten)


def main():
    """main."""
    args = get_args()
    out = args.out

    max_peaks, min_inten = args.max_peaks, args.min_inten
    num_bins, upper_limit = args.num_bins, args.upper_limit
    num_workers = args.num_workers
    form_folder = Path(args.form_folder)
    form_files = list(form_folder.glob("*.json"))

    if out is None:
        out = form_folder.parent / f"{form_folder.stem}_binned.p"

    spec_names = [i.stem.replace("pred_", "") for i in form_files]

    # Test case
    # dag_file = dag_files[0]
    # binned, root = bin_dag_file(dag_file, max_peaks=max_peaks,
    #                            upper_limit=upper_limit, num_bins=num_bins,
    #                            min_inten=min_inten)

    read_dag_file = partial(
        bin_form_file,
        max_peaks=max_peaks,
        upper_limit=upper_limit,
        num_bins=num_bins,
        min_inten=min_inten,
    )

    if num_workers > 0:
        outs = common.chunked_parallel(
            form_files,
            read_dag_file,
            max_cpu=num_workers,
        )
        binned, smis = zip(*outs)
    else:
        outs = [read_dag_file(i) for i in form_files]
        binned, smis = zip(*outs)

    binned_stack = np.concatenate(binned, 0)
    output = {
        "preds": binned_stack,
        "smiles": smis,
        "spec_names": spec_names,
        "num_bins": num_bins,
        "upper_limit": upper_limit,
    }
    with open(out, "wb") as fp:
        pickle.dump(output, fp)


if __name__ == "__main__":
    main()
