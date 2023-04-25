""" dag_to_subform.py

Convert dag folder into a subform folder

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
    parser.add_argument("--out-dir", action="store")
    parser.add_argument("--all-h-shifts", action="store_true", default=False,
                        help="If true, expand each fragment with its h shifts")
    return parser.parse_args()


def dag_file_to_form(dag_file: dict, out_dir: Path, all_h_shifts: bool=False):
    """dag_file_to_form.

    Args:
        dag_file (dict): dag_file
        out_dir (Path): out_dir
        all_h_shifts: if true, use all h shifts specified
    """
    dag = json.load(open(dag_file, "r"))
    root = dag['root_inchi']
    base_form = common.form_from_inchi(root)
    smiles = common.smiles_from_inchi(root)
    name = Path(dag_file).stem.replace("pred_", "")
    outfile = Path(out_dir) / f"{name}.json"

    frags = dag['frags']
    engine = fragmentation.FragmentEngine(mol_str=root,
                                          mol_str_type="inchi")
    forms, mz, intens = [], [], []
    for k, v in frags.items():
        base_mass = engine.single_mass(v['frag'])
        if all_h_shifts:
            max_remove, max_add = v['max_remove_hs'], v['max_add_hs']
            for shift_num in list(range(-max_remove, max_add + 1)):
                forms.append(engine.formula_from_frag(v['frag'], shift_num))
                shift_mass = engine.shift_bucket_masses[shift_num + engine.max_broken_bonds]
                mz.append(shift_mass + base_mass)
                intens.append(1)
        else:
            for shift_mass, shift_num, inten in zip(engine.shift_bucket_masses,
                                                    engine.shift_buckets,
                                                    v['intens']):
                if inten > 0:
                    forms.append(engine.formula_from_frag(v['frag'], shift_num))
                    mz.append(shift_mass + base_mass)
                    intens.append(inten)

    tbl = {"mz": mz,
           "ms2_inten": intens,
           "rel_inten": intens,
           "mono_mass": mz,
           "formula_mass_no_adduct": mz,
           "mass_diff": [0] * len(mz),
           "formula":forms,
           "ions": ["H+"] * len(mz),
           }
    output = {"cand_form": base_form, "spec_name": name, "cand_ion": "H+",
              "output_tbl": tbl, "smiles": smiles}
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=4)


def main():
    """main.
    """
    args = get_args()
    out = args.out_dir
    all_h_shifts = args.all_h_shifts

    num_workers = args.num_workers
    dag_folder = Path(args.dag_folder)
    dag_files = list(dag_folder.glob("*.json"))


    if out is None:
        out = dag_folder.parent / f"{dag_folder.stem}_subform.p"
    out = Path(out)
    out.mkdir(exist_ok=True)

    # Test case
    dag_file = dag_files[0]
    apply_fn = partial(dag_file_to_form, out_dir=out, all_h_shifts=all_h_shifts)
    if num_workers > 0:
        outs = common.chunked_parallel(dag_files,
                                       apply_fn,
                                       max_cpu=num_workers,)
    else:
        outs = [apply_fn(i) for i in dag_files]


if __name__=="__main__": 
    main()
