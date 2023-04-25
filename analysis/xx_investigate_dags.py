""" xx_investigate_dags.py

Playground to analyze / look at adgs

"""
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

    print(len(set(mz)), len(mz))

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
    return binned, root


def dag_stats(
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
    root = dag["root_inchi"]
    frags = dag["frags"]
    engine = fragmentation.FragmentEngine(mol_str=root, mol_str_type="inchi")
    subforms = []
    for i in list(dag["frags"].values()):
        # subforms.append(i['form'])
        subforms.append(i["base_mass"])

    print(f"Len of subforms: {len(subforms)}")
    print(f"Len of unique subforms: {len(set(subforms))}")
    if (len((set(subforms))) - len(subforms)) < -5:
        import pdb

        pdb.set_trace()

    # mz, intens = [], []
    # for k, v in frags.items():
    #    base_mass = engine.single_mass(v['frag'])
    #    new_masses = engine.shift_bucket_masses + base_mass
    #    new_intens = v['intens']
    #    mz.extend(new_masses.tolist())
    #    intens.extend(new_intens)

    ## Cap at min
    # mz, intens = np.array(mz), np.array(intens)
    # min_inten_mask = intens > min_inten
    # mz = mz[min_inten_mask]
    # intens = intens[min_inten_mask]

    ## Keep 100 max
    # argsort_intens = np.argsort(intens)[::-1][:max_peaks]
    # mz = mz[argsort_intens]
    # intens = intens[argsort_intens]
    # spec_ar = np.vstack([mz, intens]).transpose(1, 0)

    ## Bin intensities
    # binned = common.bin_spectra([spec_ar], num_bins, upper_limit,
    #                            pool_fn="max")
    ##


def main():
    """main."""
    args = get_args()

    max_peaks, min_inten = args.max_peaks, args.min_inten
    num_bins, upper_limit = args.num_bins, args.upper_limit
    dag_folder = Path(args.dag_folder)
    dag_files = list(dag_folder.glob("*.json"))

    read_dag_file = partial(
        dag_stats,
        max_peaks=max_peaks,
        upper_limit=upper_limit,
        num_bins=num_bins,
        min_inten=min_inten,
    )

    dag_outs = [read_dag_file(i) for i in dag_files]


if __name__ == "__main__":
    main()
