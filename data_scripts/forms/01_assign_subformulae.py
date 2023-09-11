""" assign_subformulae.py

Given a set of spectra and candidates from a labels file, assign subformulae and save to JSON files.

"""

from pathlib import Path
import argparse
from functools import partial
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from ms_pred import common
from ms_pred.magma import frag_subform


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="data/spec_datasets/gnps2015_debug", help="Data debug"
    )
    parser.add_argument(
        "--output-dir-name",
        default=None,
        help="Name of output dir (in data_dir/subformulae)",
    )
    parser.add_argument(
        "--labels-file",
        default="data/spec_datasets/gnps2015_debug/labels.tsv",
        help="Data debug",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug flag."
    )
    parser.add_argument(
        "--use-all",
        action="store_true",
        default=False,
        help="If true, do not subset formula.",
    )
    parser.add_argument(
        "--use-magma",
        action="store_true",
        default=False,
        help="If true, do not subset formula.",
    )
    parser.add_argument(
        "--mass-diff-type",
        default="ppm",
        type=str,
        help="Type of mass difference - absolute differece (abs) or relative difference (ppm).",
    )
    parser.add_argument(
        "--mass-diff-thresh",
        action="store",
        default=10,
        type=float,
        help="Threshold of mass difference.",
    )
    parser.add_argument(
        "--inten-thresh",
        action="store",
        default=0.001,
        type=float,
        help="Threshold of MS2 subpeak intensity (normalized to 1).",
    )
    parser.add_argument(
        "--max-formulae",
        action="store",
        default=50,
        type=int,
        help="Max number of peaks to keep",
    )
    return parser.parse_args()


def get_output_dict(
    spec_name: str,
    spec: np.ndarray,
    form: str,
    mass_diff_type: str,
    mass_diff_thresh: float,
    inten_thresh: float,
    adduct_type: str,
    max_formulae: int = 100,
    use_all=False,
    smiles: str = "",
    use_magma: bool = False,
) -> dict:
    """get_output_dict.
    This function attemps to take an array of mass intensity values and assign
    formula subsets to subpeaks
    Args:
        spec (np.ndarray): spec
        form (str): form
        abs_mass_diff (float): abs_mass_diff
        inten_thresh (float): Intensity threshold
        max_formulae (int): Max formulae to output
        use_all (bool): If true, don't subset to formula
        smiles (str): Smiles string
        use_magma (bool): If true, use the magma algorithm

    Returns:
        Dict
    """
    # This is the case for some erroneous MS2 files for which proc_spec_file return None
    # All the MS2 subpeaks in these erroneous MS2 files has mz larger than parentmass
    if spec is None:
        output_dict = {
            "cand_form": form,
            "spec_name": spec_name,
            "cand_ion": adduct_type,
            "output_tbl": None,
        }
        return output_dict

    # Filter down
    spec = common.max_inten_spec(spec, max_formulae, inten_thresh=inten_thresh)
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]
    adduct_masses = common.ion2mass[adduct_type]

    if use_all:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "rel_inten": list(spec_intens),  # rel_inten),
            "mono_mass": list(spec_masses),
            "formula_mass_no_adduct": list(spec_masses - adduct_masses),
            "mass_diff": [0] * len(spec_masses),
            "formula": [""] * len(spec_masses),
            "ions": [adduct_masses] * len(spec_masses),
        }

        if len(spec_intens) == 0:
            output_tbl = None
        output_dict = {
            "cand_form": form,
            "spec_name": spec_name,
            "cand_ion": adduct_type,
            "output_tbl": output_tbl,
        }
        return output_dict

    if use_magma:

        # TODO: Replace with new magma generator
        fe = frag_subform.FragmentEngine(
            mol_str=smiles,
            # max_tree_depth=3,
        )
        try:
            fe.generate_fragments()
        except:
            print(f"Error with generating fragments for spec {smiles}")
            return {
                "cand_form": form,
                "spec_name": spec_name,
                "cand_ion": adduct_type,
                "output_tbl": None,
            }
        cross_prod, masses = fe.get_frag_forms()
    else:
        cross_prod, masses = common.get_all_subsets(form)
    masses_with_adduct = masses + adduct_masses
    adduct_types = np.array([adduct_type] * len(masses_with_adduct))
    mass_diffs = np.abs(spec_masses[:, None] - masses_with_adduct[None, :])

    formula_inds = mass_diffs.argmin(-1)
    min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]

    if mass_diff_type == "ppm":
        mass_divisior = np.copy(spec_masses)
        mass_divisior[mass_divisior <= 200] = 200
        min_mass_diff = (min_mass_diff / mass_divisior) * 1e6
    elif mass_diff_type == "abs":
        pass

    # Filter by abs mass diff
    valid_mask = min_mass_diff < mass_diff_thresh
    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    min_mass_diff = min_mass_diff[valid_mask]
    formula_inds = formula_inds[valid_mask]

    formulas = np.array([common.vec_to_formula(j) for j in cross_prod[formula_inds]])
    formula_masses = masses_with_adduct[formula_inds]
    formula_mass_no_adduct = masses[formula_inds]
    adduct_types = adduct_types[formula_inds]

    # Build mask for uniqueness on formula and adduct
    # note that adduct are all the same for one subformula assignment
    # hence we only need to consider the uniqueness of the formula
    formula_idx_dict = {}
    uniq_mask = []
    for idx in range(len(formulas)):
        formula = formulas[idx]
        if formula not in formula_idx_dict:
            uniq_mask.append(True)
            formula_idx_dict[formula] = idx
        else:
            merge_idx = formula_idx_dict[formula]
            uniq_mask.append(False)
            spec_intens[merge_idx] = max(spec_intens[idx], spec_intens[merge_idx])

    spec_masses = spec_masses[uniq_mask]
    spec_intens = spec_intens[uniq_mask]
    min_mass_diff = min_mass_diff[uniq_mask]
    formula_masses = formula_masses[uniq_mask]
    formula_mass_no_adduct = formula_mass_no_adduct[uniq_mask]
    formulas = formulas[uniq_mask]
    adduct_types = adduct_types[uniq_mask]

    # Renormalize
    # to calculate explained intensity, let's preserve the original normalized intensity
    if spec_intens.size == 0:
        output_dict = {
            "cand_form": form,
            "spec_name": spec_name,
            "cand_ion": adduct_type,
            "output_tbl": None,
        }
    else:
        # if mass_diff_type = ppm, then mass_diff is calculated using ppm
        # if mass_diff_type = abs, then mass_diff is in the unit of Dalton

        # Use rel inten, but assume it's already been processed
        # rel_inten = spec_intens / (spec_intens.max())
        # rel_inten = np.sqrt(rel_inten)
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "rel_inten": list(spec_intens),  # rel_inten),
            "mono_mass": list(formula_masses),
            "formula_mass_no_adduct": list(formula_mass_no_adduct),
            "mass_diff": list(min_mass_diff),
            "formula": list(formulas),
            "ions": [adduct for adduct in list(adduct_types)],
        }

        output_dict = {
            "cand_form": form,
            "spec_name": spec_name,
            "cand_ion": adduct_type,
            "output_tbl": output_tbl,
        }
    return output_dict


def process_spec_file(spec_name: str, data_dir: str):
    """process_spec_file.
    Args:
        spec_name (str): spec_name
        data_dir (str): data_dir
    """
    spec_file = data_dir / "spec_files" / f"{spec_name}.ms"
    meta, tuples = common.parse_spectra(spec_file)
    spec = common.process_spec_file(meta, tuples)
    return spec_name, spec


def main():
    """main."""

    args = get_args()
    data_dir = Path(args.data_dir)
    label_path = Path(args.labels_file)
    mass_diff_thresh = args.mass_diff_thresh
    mass_diff_type = args.mass_diff_type
    inten_thresh = args.inten_thresh
    use_all = args.use_all

    max_form = args.max_formulae
    max_formulae = args.max_formulae
    debug = args.debug
    use_magma = args.use_magma

    # Read in labels
    labels_df = pd.read_csv(label_path, sep="\t")
    if debug:
        labels_df = labels_df[:50]

    # Define ooutput directory name
    subform_dir = data_dir / "subformulae"
    subform_dir.mkdir(exist_ok=True)
    output_dir_name = args.output_dir_name
    if output_dir_name is None:
        output_dir_name = f"subform_{max_form}"
    output_dir = subform_dir / output_dir_name
    output_dir.mkdir(exist_ok=True)

    spec_fn_lst = labels_df["spec"].to_list()
    proc_spec_full = partial(process_spec_file, data_dir=data_dir)
    # input_specs = [proc_spec_full(i) for i in tqdm(spec_fn_lst)]
    input_specs = common.chunked_parallel(spec_fn_lst, proc_spec_full, chunks=100)

    # input_specs contains a list of tuples (spec, subpeak tuple array)
    input_specs_dict = {tup[0]: tup[1] for tup in input_specs}

    export_dicts = []
    for _, row in labels_df.iterrows():
        spec = row["spec"]
        new_entry = {
            "spec": input_specs_dict[spec],
            "form": row["formula"],
            "mass_diff_type": mass_diff_type,
            "spec_name": spec,
            "mass_diff_thresh": mass_diff_thresh,
            "inten_thresh": inten_thresh,
            "max_formulae": max_formulae,
            "adduct_type": row["ionization"],
            "use_all": use_all,
            "smiles": row["smiles"],
            "use_magma": use_magma,
        }
        export_dicts.append(new_entry)

    # Build dicts
    print(f"There are {len(export_dicts)} spec-cand pairs this spec files")
    export_wrapper = lambda x: get_output_dict(**x)

    if debug:
        output_dict_lst = [export_wrapper(i) for i in export_dicts[:10]]
    else:

        output_dict_lst = common.chunked_parallel(
            export_dicts, export_wrapper, chunks=100
        )
    assert len(export_dicts) == len(output_dict_lst)

    # Write all output jsons to files
    for output_dict in tqdm(output_dict_lst):
        with open(output_dir / f'{output_dict["spec_name"]}.json', "w") as f:
            json.dump(output_dict, f, indent=4)
            f.close()


if __name__ == "__main__":
    main()
