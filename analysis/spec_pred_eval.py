""" spec_pred_eval.py

Use to compare binned predictions to ground truth spec values

"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import yaml
import pickle
from collections import defaultdict
from functools import partial
from numpy.linalg import norm

import ms_pred.common as common


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="canopus_train_public")
    parser.add_argument("--formula-dir-name", default="subform_20")
    parser.add_argument(
        "--binned-pred-file", default="results/2022_12_18_ffn_pred/fp_preds.p"
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument(
        "--min-inten",
        type=float,
        default=1e-5,
        help="Minimum intensity to call a peak in prediction",
    )
    parser.add_argument(
        "--max-peaks", type=int, default=20, help="Max num peaks to call"  # 20,
    )
    return parser.parse_args()


def process_spec_file(spec_name, num_bins: int, upper_limit: int, spec_dir: Path):
    """process_spec_file."""
    spec_file = spec_dir / f"{spec_name}.json"
    loaded_json = json.load(open(spec_file, "r"))

    if loaded_json.get("output_tbl") is None:
        return None

    # Load without adduct involved
    mz = loaded_json["output_tbl"]["formula_mass_no_adduct"]
    inten = loaded_json["output_tbl"]["ms2_inten"]
    spec_ar = np.vstack([mz, inten]).transpose(1, 0)
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit)
    avged = binned[0]

    # normed = common.norm_spectrum(binned)
    # avged = normed.mean(0)
    return avged


def main(args):
    """main."""
    dataset = args.dataset
    formula_dir_name = args.formula_dir_name
    data_folder = Path(f"data/spec_datasets/{dataset}/subformulae/{formula_dir_name}/")
    min_inten = args.min_inten
    max_peaks = args.max_peaks

    binned_pred_file = Path(args.binned_pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = binned_pred_file.parent / "pred_eval.yaml"
        outfile_grouped = binned_pred_file.parent / "pred_eval_grouped.tsv"

    pred_specs = pickle.load(open(binned_pred_file, "rb"))
    pred_spec_ars = pred_specs["preds"]
    pred_smiles = pred_specs["smiles"]
    pred_spec_names = pred_specs["spec_names"]
    upper_limit = pred_specs["upper_limit"]
    num_bins = pred_specs["num_bins"]

    read_spec = partial(
        process_spec_file,
        num_bins=num_bins,
        upper_limit=upper_limit,
        spec_dir=data_folder,
    )
    true_specs = common.chunked_parallel(
        pred_spec_names, read_spec, chunks=100, max_cpu=16, timeout=4000, max_retries=3
    )
    running_lists = defaultdict(lambda: [])
    output_entries = []
    for pred_ar, pred_smi, pred_spec, true_spec in zip(
        pred_spec_ars, pred_smiles, pred_spec_names, true_specs
    ):
        if true_spec is None:
            continue

        # Don't norm spec
        ## Norm pred spec by max
        # if np.max(pred_ar) > 0:
        #    pred_ar = np.array(pred_ar) / np.max(pred_ar)

        # Get all actual bins
        pred_greater = np.argwhere(pred_ar > min_inten).flatten()
        pos_bins_sorted = sorted(pred_greater, key=lambda x: -pred_ar[x])
        pos_bins = pos_bins_sorted[:max_peaks]

        new_pred = np.zeros_like(pred_ar)
        new_pred[pos_bins] = pred_ar[pos_bins]
        pred_ar = new_pred

        norm_pred = max(norm(pred_ar), 1e-6)
        norm_true = max(norm(true_spec), 1e-6)
        cos_sim = np.dot(pred_ar, true_spec) / (norm_pred * norm_true)
        mse = np.mean((pred_ar - true_spec) ** 2)

        # Compute validity

        # Get all possible bins that would be valid
        if pred_smi is not None:
            true_form = common.form_from_smi(pred_smi)
            _cross_prod, masses = common.get_all_subsets(true_form)
            possible = common.digitize_ar(
                masses, num_bins=num_bins, upper_limit=upper_limit
            )
            smiles_mass = common.mass_from_smi(pred_smi)
            ikey = common.inchikey_from_smiles(pred_smi)
        else:
            possible = []
            smiles_mass = 0
            ikey = ""

        possible_set = set(possible)
        pred_set = set(pos_bins)
        overlap = pred_set.intersection(possible_set)

        if len(pred_set) == 0:
            frac_valid = 1.0
        else:
            frac_valid = len(overlap) / len(pred_set)

        # Bin 2420 from pred set
        # 2421 in the possible set, grrr. Must be a boundary decision?
        # nist_3143557
        # mass n;umber 3854, masss 242.01614gt

        # Compute true overlap
        true_inds = np.argwhere(true_spec > min_inten).flatten()
        true_bins_sorted = sorted(true_inds, key=lambda x: -true_spec[x])
        true_bins = set(true_bins_sorted[:max_peaks])

        total_covered = true_bins.intersection(pos_bins)
        overlap_coeff = len(total_covered) / max(
            min(len(true_bins), len(pos_bins)), 1e-6
        )
        coverage = len(total_covered) / max(len(true_bins), 1e-6)

        # Load true spec
        output_entry = {
            "name": pred_spec,
            "inchi": ikey,
            "cos_sim": float(cos_sim),
            "mse": float(mse),
            "frac_valid": float(frac_valid),
            "overlap_coeff": float(overlap_coeff),
            "coverage": float(coverage),
            "len_targ": len(true_bins),
            "len_pred": len(pos_bins),
            "compound_mass": smiles_mass,
            "mass_bin": common.bin_mass_results(smiles_mass),
        }

        output_entries.append(output_entry)
        running_lists["cos_sim"].append(cos_sim)
        running_lists["mse"].append(mse)
        running_lists["frac_valid"].append(frac_valid)
        running_lists["overlap_coeff"].append(overlap_coeff)
        running_lists["coverage"].append(coverage)
        running_lists["len_targ"].append(len(true_bins))
        running_lists["len_pred"].append(len(pos_bins))

    final_output = {
        "dataset": dataset,
        "data_folder": str(data_folder),
        "individuals": sorted(output_entries, key=lambda x: x["cos_sim"]),
    }

    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v))

    df = pd.DataFrame(output_entries)
    df_grouped = pd.concat(
        [df.groupby("mass_bin").mean(), df.groupby("mass_bin").size()], 1
    )
    df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

    all_mean = df.mean()
    all_mean["num_examples"] = len(df)
    all_mean.name = "avg"
    df_grouped = df_grouped.append(all_mean)
    df_grouped.to_csv(outfile_grouped, sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2)
        print(out_str)
        fp.write(out_str)


if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
