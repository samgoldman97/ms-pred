from pathlib import Path
import json
import argparse
import yaml
import pickle
from tqdm import tqdm
from collections import defaultdict
from functools import partial

import numpy as np
from numpy.linalg import norm

import pandas as pd

import ms_pred.common as common


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="canopus_train_public")
    parser.add_argument("--formula-dir-name", default="subform_20")
    parser.add_argument(
        "--binned-pred-file",
        default="results/ffn_baseline_cos/retrieval/split_1/fp_preds.p",
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--dist-fn", default="cos")
    return parser.parse_args()


def process_spec_file(spec_name, num_bins: int, upper_limit: int, spec_dir: Path):
    """process_spec_file."""
    spec_file = spec_dir / f"{spec_name}.json"
    if not spec_file.exists():
        print(f"Cannot find spec {spec_file}")
        return np.zeros(num_bins)
    loaded_json = json.load(open(spec_file, "r"))

    if loaded_json.get("output_tbl") is None:
        return None

    # Load without adduct involved
    mz = loaded_json["output_tbl"]["formula_mass_no_adduct"]
    inten = loaded_json["output_tbl"]["ms2_inten"]
    spec_ar = np.vstack([mz, inten]).transpose(1, 0)
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit)
    avged = binned[0]
    return avged


def cos_dist(cand_preds, true_spec) -> np.ndarray:
    """cos_dist.

    Args:
        cand_preds:
        true_spec:

    Returns:
        np.ndarray:
    """

    # Assume sparse
    pred_specs = np.zeros((cand_preds.shape[0], true_spec.shape[0]))
    inds = cand_preds[:, :, 0].astype(int)
    pos_1 = np.ones(inds.shape) * np.arange(inds.shape[0])[:, None]
    pred_specs[pos_1.flatten().astype(int), inds.flatten()] = cand_preds[
        :, :, 1
    ].flatten()

    # TEMP
    # cand_preds = cand_preds[:, :10, :]
    norm_pred = norm(pred_specs, axis=-1)  # , ord=2) #+ 1e-22
    norm_true = norm(true_spec, axis=-1)  # , ord=2) #+ 1e-22

    # Cos
    dist = 1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true)
    return dist


def rank_test_entry(
    cand_ikeys,
    cand_preds,
    true_spec,
    true_ikey,
    spec_name,
    true_smiles,
    dist_fn="cos",
    **kwargs,
):
    """rank_test_entry.

    Args:
        cand_ikeys:
        cand_preds:
        true_spec:
        true_ikey:
        spec_name:
        true_smiles:
        kwargs:
    """
    if dist_fn == "cos":
        dist = cos_dist(cand_preds=cand_preds, true_spec=true_spec)
    elif dist_fn == "random":
        dist = np.random.randn(cand_preds.shape[0])
    else:
        raise NotImplementedError()

    true_ind = np.argwhere(cand_ikeys == true_ikey).flatten()

    # Now need to find which position 0 is in  --> should be 28th
    resorted = np.argsort(dist)
    # inds_found = np.argsort(resorted)

    # resorted_dist = dist[resorted]
    # NOTE: resorted is out of bounds
    resorted_ikeys = cand_ikeys[resorted]
    resorted_dist = dist[resorted]
    
    top_hit = str(resorted_ikeys[0]) if len(resorted_ikeys) > 0 else None

    true_mass = common.mass_from_smi(true_smiles)
    mass_bin = common.bin_mass_results(true_mass)

    #assert len(true_ind) == 1
    if len(true_ind) == 0:
        print(f"Could not find true ind for {spec_name}")
        return {
            "ind_recovered": 1e10,
            "total_decoys": len(resorted_ikeys),
            "mass": float(true_mass),
            "mass_bin": mass_bin,
            "true_dist": None,
            "spec_name": str(spec_name),
            "top_hit": top_hit,
        }
    elif len(true_ind) > 1:
        raise ValueError()
    else:
        true_ind = true_ind[0]
        true_dist = dist[true_ind]

        # ind_found = inds_found[true_ind]
        # tie_shift = np.sum(true_dist == dist) - 1
        # ind_found_init = ind_found
        # ind_found = ind_found + tie_shift
        ind_found = np.argwhere(resorted_dist == true_dist).flatten()[-1]

        # Add 1 in case it was first to be top 1 not zero
        ind_found = ind_found + 1

        return {
            "ind_recovered": float(ind_found),
            "total_decoys": len(resorted_ikeys),
            "mass": float(true_mass),
            "mass_bin": mass_bin,
            "true_dist": float(true_dist),
            "spec_name": str(spec_name),
            "top_hit": top_hit,
        }


def main(args):
    """main."""
    dataset = args.dataset
    formula_dir_name = args.formula_dir_name
    dist_fn = args.dist_fn
    data_folder = Path(f"data/spec_datasets/{dataset}")
    form_folder = data_folder / f"subformulae/{formula_dir_name}/"
    data_df = pd.read_csv(data_folder / "labels.tsv", sep="\t")
    data_df['spec'] = [str(i) for i in data_df['spec']]

    name_to_ikey = dict(data_df[["spec", "inchikey"]].values)
    name_to_smi = dict(data_df[["spec", "smiles"]].values)
    name_to_ion = dict(data_df[["spec", "ionization"]].values)

    binned_pred_file = Path(args.binned_pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = binned_pred_file.parent / f"rerank_eval_{dist_fn}.yaml"
        outfile_grouped_ion = (
            binned_pred_file.parent / f"rerank_eval_grouped_ion_{dist_fn}.tsv"
        )
        outfile_grouped_mass = (
            binned_pred_file.parent / f"rerank_eval_grouped_mass_{dist_fn}.tsv"
        )
    else:
        outfile = Path(outfile)
        outfile_grouped_ion = outfile.parent / f"{outfile.stem}_grouped_ion.tsv"
        outfile_grouped_mass = outfile.parent / f"{outfile.stem}_grouped_mass.tsv"

    pred_specs = pickle.load(open(binned_pred_file, "rb"))
    pred_spec_ars = pred_specs["preds"]
    # pred_smiles = np.array(pred_specs['smiles'])
    pred_ikeys = np.array(pred_specs["ikeys"])
    pred_spec_names = np.array(pred_specs["spec_names"], dtype=str)
    pred_spec_names_unique = [str(i) for i in np.unique(pred_spec_names)]
    upper_limit = pred_specs["upper_limit"]
    num_bins = pred_specs["num_bins"]
    use_sparse = pred_specs["sparse_out"]

    # Only use sparse valid for now
    assert use_sparse

    read_spec = partial(
        process_spec_file,
        num_bins=num_bins,
        upper_limit=upper_limit,
        spec_dir=form_folder,
    )
    true_specs = common.chunked_parallel(
        pred_spec_names_unique,
        read_spec,
        chunks=100,
        max_cpu=16,
        timeout=4000,
        max_retries=3,
    )
    name_to_spec = dict(zip(pred_spec_names_unique, true_specs))

    # Create a list of dicts, bucket by mass, etc.
    all_entries = []
    for spec_name in tqdm(pred_spec_names_unique):
        spec_name = str(spec_name)

        # Get candidates
        bool_sel = pred_spec_names == spec_name
        cand_ikeys = pred_ikeys[bool_sel]
        cand_preds = pred_spec_ars[bool_sel]
        true_spec = name_to_spec[spec_name]
        true_smi = name_to_smi[spec_name]
        new_entry = {
            "cand_ikeys": cand_ikeys,
            "cand_preds": cand_preds,
            "true_spec": true_spec,
            "true_smiles": true_smi,
            "true_ikey": name_to_ikey[spec_name],
            "spec_name": spec_name,
        }

        if true_spec is None:
            continue
        all_entries.append(new_entry)

    rank_test_entry_ = partial(rank_test_entry, dist_fn=dist_fn)
    all_out = [rank_test_entry_(**test_entry) for test_entry in all_entries]

    # Compute avg and individual stats
    k_vals = list(range(1, 11))
    running_lists = defaultdict(lambda: [])
    output_entries = []
    for out in all_out:
        output_entries.append(out)
        for k in k_vals:
            below_k = out["ind_recovered"] is not None and out["ind_recovered"] <= k
            running_lists[f"top_{k}"].append(below_k)
            out[f"top_{k}"] = below_k
        running_lists["total_decoys"].append(out["total_decoys"])
        running_lists["true_dist"].append(out["true_dist"])

    final_output = {
        "dataset": dataset,
        "data_folder": str(data_folder),
        "dist_fn": dist_fn,
        "individuals": sorted(output_entries, key=lambda x: x["ind_recovered"]
                              if x['ind_recovered'] is not None else -99999),
    }

    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v))

    for i in output_entries:
        i["ion"] = name_to_ion[i["spec_name"]]

    df = pd.DataFrame(output_entries)

    for group_key, out_name in zip(
        ["mass_bin", "ion"], [outfile_grouped_mass, outfile_grouped_ion]
    ):
        df_grouped = pd.concat(
            [df.groupby(group_key).mean(numeric_only=True), df.groupby(group_key).size()], axis=1
        )
        df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

        all_mean = df.mean(numeric_only=True)
        all_mean["num_examples"] = len(df)
        all_mean.name = "avg"
        df_grouped = pd.concat([df_grouped, all_mean.to_frame().T], axis=0)
        df_grouped.to_csv(out_name, sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2)
        print(out_str)
        fp.write(out_str)


if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
