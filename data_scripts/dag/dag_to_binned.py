""" Convert dag folder into a binned spec file """
import json
import yaml
import argparse
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--dag-folder", action="store")
    parser.add_argument(
        "--min-inten",
        type=float,
        default=0,
        help="Minimum intensity to call a peak in prediction",
    )
    parser.add_argument(
        "--max-peaks", type=int, default=100, help="Max num peaks to call"
    )
    parser.add_argument("--num-bins", type=int, default=3000, help="Num bins")
    parser.add_argument("--upper-limit", type=int, default=1500, help="upper lim")
    parser.add_argument("--out", action="store")
    return parser.parse_args()


def bin_dag(
    dag: dict, max_peaks: int, upper_limit: int, num_bins: int, min_inten: float
) -> np.ndarray:
    """bin_dag.

    Args:
        dag (dict): dag
        max_peaks (int): max_peaks
        upper_limit (int): upper_limit
        num_bins (int): num_bins
        min_inten (float): min_inten

    Returns:
        np.ndarray:
    """
    root = dag["root_inchi"]
    frags = dag["frags"]
    engine = fragmentation.FragmentEngine(mol_str=root, mol_str_type="inchi")

    mz, intens = [], []
    for k, v in frags.items():
        base_mass = engine.single_mass(v["frag"])
        new_masses = engine.shift_bucket_masses + base_mass
        new_intens = v["intens"]
        mz.extend(new_masses.tolist())
        intens.extend(new_intens)

    # Cap at min
    mz, intens = np.array(mz), np.array(intens)
    min_inten_mask = intens > min_inten
    mz = mz[min_inten_mask]
    intens = intens[min_inten_mask]

    # Bin intensities
    # Keep 100 max and above thresh
    spec_ar = np.vstack([mz, intens]).transpose(1, 0)
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit, pool_fn="max")
    assert binned.shape[0] == 1
    binned = binned[0]

    pred_greater = np.argwhere(binned > min_inten).flatten()
    pos_bins_sorted = sorted(pred_greater, key=lambda x: -binned[x])
    pos_bins = pos_bins_sorted[:max_peaks]

    new_pred = np.zeros_like(binned)
    new_pred[pos_bins] = binned[pos_bins]
    binned = new_pred
    binned = binned[None, :]

    return binned, root


def bin_dag_file(
    dag_file: dict, max_peaks: int, upper_limit: int, num_bins: int, min_inten: float
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
    dag = json.load(open(dag_file, "r"))
    return bin_dag(dag, max_peaks, upper_limit, num_bins, min_inten)


def main():
    """main."""
    args = get_args()
    out = args.out

    max_peaks, min_inten = args.max_peaks, args.min_inten
    num_bins, upper_limit = args.num_bins, args.upper_limit
    num_workers = args.num_workers
    dag_folder = Path(args.dag_folder)
    dag_files = list(dag_folder.glob("*.json"))

    if out is None:
        out = dag_folder.parent / f"{dag_folder.stem}_binned.p"

    spec_names = [i.stem.replace("pred_", "") for i in dag_files]

    # Test case
    # dag_file = dag_files[0]
    # binned, root = bin_dag_file(dag_file, max_peaks=max_peaks,
    #                            upper_limit=upper_limit, num_bins=num_bins,
    #                            min_inten=min_inten)

    read_dag_file = partial(
        bin_dag_file,
        max_peaks=max_peaks,
        upper_limit=upper_limit,
        num_bins=num_bins,
        min_inten=min_inten,
    )

    if num_workers > 0:
        outs = common.chunked_parallel(
            dag_files,
            read_dag_file,
            max_cpu=num_workers,
        )
        binned, inchis = zip(*outs)
        smis = common.chunked_parallel(
            inchis, common.smiles_from_inchi, max_cpu=num_workers
        )
    else:
        outs = [read_dag_file(i) for i in dag_files]
        binned, inchis = zip(*outs)
        smis = [common.smiles_from_inchi(i) for i in inchis]

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
