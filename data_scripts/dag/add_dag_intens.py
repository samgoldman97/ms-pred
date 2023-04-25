""" 05_add_dag_intens.py

Given a set of predicted dags, add intensities to them from the gold standard

"""
import json
import argparse
import copy
from pathlib import Path

import ms_pred.magma.run_magma as run_magma
import ms_pred.common as common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--pred-dag-folder", action="store")
    parser.add_argument("--true-dag-folder", action="store")
    parser.add_argument("--out-dag-folder", action="store", default=None)
    parser.add_argument("--add-raw", action="store_true", default=False,)
    return parser.parse_args()


def relabel_tree(pred_dag_file: Path, true_dag_file: Path, out_dag_file: Path,
                 max_bonds: int, add_raw: bool) -> None:
    """relabel_tree.

    Args:
        pred_dag_file (Path): pred_dag_file
        true_dag_file (Path): true_dag_file
        out_dag_file (Path): out_dag_file
        max_bonds (int): max_bonds
        add_raw

    Returns:
        None:
    """
    zero_vec = [0] * (2 * max_bonds + 1)
    if not true_dag_file.exists():
        return

    pred_dag = json.load(open(pred_dag_file, "r"))
    true_dag = json.load(open(true_dag_file, "r"))
    if add_raw:
        true_tbl = true_dag['output_tbl']
        raw_spec = list(zip(true_tbl['formula_mass_no_adduct'],
                            true_tbl['rel_inten']))
        pred_dag['raw_spec'] = raw_spec
    else:
        pred_frags, true_frags = pred_dag['frags'], true_dag['frags']

        for k, pred_frag in pred_frags.items():
            if k in true_frags:
                true_frag = true_frags[k]
                pred_frag['intens'] = true_frag['intens']
            else:
                pred_frag['intens'] = copy.deepcopy(zero_vec)
    

    with open(out_dag_file, "w") as fp:
        json.dump(pred_dag, fp, indent=2)


def main():
    """main.
    """
    args = get_args()
    pred_dag_folder = Path(args.pred_dag_folder)
    true_dag_folder = Path(args.true_dag_folder)
    out_dag_folder = args.out_dag_folder
    add_raw = args.add_raw

    if out_dag_folder is None:
        out_dag_folder = pred_dag_folder
    out_dag_folder = Path(out_dag_folder)
    out_dag_folder.mkdir(exist_ok=True)

    max_bonds = run_magma.FRAGMENT_ENGINE_PARAMS['max_broken_bonds']

    num_workers = args.num_workers
    pred_dag_files = list(pred_dag_folder.glob("*.json"))
    true_dag_files = [true_dag_folder / i.name.replace("pred_", "")
                      for i in pred_dag_files]
    out_dag_files = [out_dag_folder / i.name for i in pred_dag_files]
    arg_dicts = [
        {"pred_dag_file": i,
         "true_dag_file": j,
         "out_dag_file": k,
         "max_bonds": max_bonds,
         "add_raw": add_raw,
         }
        for i, j, k in zip(pred_dag_files, true_dag_files, out_dag_files)
    ]

    # Run
    wrapper_fn = lambda arg_dict: relabel_tree(**arg_dict)
    if num_workers == 0:
        [wrapper_fn(i) for i in arg_dicts]
    else:
        common.chunked_parallel(arg_dicts, wrapper_fn, max_cpu=num_workers)


if __name__=="__main__": 
    main()
