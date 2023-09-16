""" Make predictions with binned and eval """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import json
from ms_pred import common
import argparse


def extract_cfm_file(spectra_file, out_dir, max_node):
    """extract_cfm_file.

    Args:
        spectra_file:
        out_dir:
        max_node:
    """
    input_name = spectra_file.stem
    meta, cfm_parsed_max = common.parse_cfm_out(spectra_file, max_merge=True)
    cfm_parsed_max_form = cfm_parsed_max.groupby("form_no_h").max().reset_index()
    cfm_parsed_max_form["inten"] /= cfm_parsed_max_form["inten"].max()
    cfm_parsed_max_form = cfm_parsed_max_form.sort_values(
        "inten", ascending=False
    ).reset_index(drop=True)
    cfm_parsed_max_form = cfm_parsed_max_form[:max_node]

    list_wrap = lambda x: x.values.tolist()
    output_tbl = {
        "ms2_inten": list_wrap(cfm_parsed_max_form["inten"]),
        "rel_inten": list_wrap(cfm_parsed_max_form["inten"]),
        "log_prob": None,
        "formula": list_wrap(cfm_parsed_max_form["form_no_h"]),
        "formula_mass_no_adduct": list_wrap(
            cfm_parsed_max_form["formula_mass_no_adduct"]
        ),
    }
    json_out = {
        "smiles": meta["SMILES"],
        "spec_name": input_name,
        "output_tbl": output_tbl,
        "cand_form": meta["Formula"],
    }

    out_file = out_dir / f"{input_name}.json"
    with open(out_file, "w") as fp:
        json.dump(json_out, fp, indent=2)


datasets = ["canopus_train_public", "nist20"]
max_nodes = [10, 20, 30, 40, 50, 100, 200]
max_node = 100
subform_name = "magma_subform_50"
split_override = None
splits = ["split_1", "scaffold_1"]

for dataset in datasets:
        res_folder = Path(f"results/cfm_id_{dataset}/")
        cfm_output_specs = res_folder / "cfm_out"
        all_files = list(cfm_output_specs.glob("*.log"))

        # Create full spec
        pred_dir_folders = []
        for split in splits:
            split_file = f"data/spec_datasets/{dataset}/splits/{split}.tsv"
            if not Path(split_file).exists():
                print(f"Skipping {split} for {dataset} due to file not found")
                continue

            split_df = pd.read_csv(split_file, sep="\t")
            save_dir = res_folder / f"{split}"
            save_dir.mkdir(exist_ok=True)

            pred_dir = save_dir / "preds/"
            pred_dir.mkdir(exist_ok=True)

            export_dir = pred_dir / "form_preds"
            export_dir.mkdir(exist_ok=True)
            test_specs = set(split_df[split_df["Fold_0"] == "test"]["spec"].values)

            to_export = [i for i in all_files if i.stem in test_specs]
            export_fn = lambda x: extract_cfm_file(x, export_dir, max_node=max_node)
            common.chunked_parallel(to_export, export_fn)
            pred_dir_folders.append(export_dir)

            # Convert all preds to binned

            # Convert to binned files
            out_binned = pred_dir / "binned_preds.p"
            cmd = f"""python data_scripts/forms/02_form_to_binned.py \\
            --max-peaks 1000 \\
            --num-bins 15000 \\
            --upper-limit 1500 \\
            --form-folder {export_dir} \\
            --num-workers 16 \\
            --out {out_binned} """
            cmd = f"{cmd}"
            print(cmd + "\n")
            subprocess.run(cmd, shell=True)

            # Eval binned preds
            eval_cmd = f"""python analysis/spec_pred_eval.py \\
            --binned-pred-file {out_binned} \\
            --max-peaks 100 \\
            --min-inten 0 \\
            --formula-dir-name no_subform \\
            --dataset {dataset}  \\
            """
            print(eval_cmd)
            subprocess.run(eval_cmd, shell=True)
