""" Sweep gen thresh """
import yaml
import pandas as pd
from pathlib import Path
import subprocess

workers = 32
devices = ",".join([])
python_file = "src/ms_pred/dag_pred/predict_gen.py"
max_nodes = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000]
subform_name = "magma_subform_50"
debug = False

res_entries = [
    {"folder": "results/dag_nist20/scaffold_1/", 
     "dataset": "nist20",
     "test_split": "scaffold_1"},

    {"folder": "results/dag_nist20/split_1_rnd1/", 
     "dataset": "nist20",
     "test_split": "split_1"},

    {"folder": "results/dag_nist20/split_1_rnd2/", 
     "dataset": "nist20",
     "test_split": "split_1"},

    {"folder": "results/dag_nist20/split_1_rnd3/", 
     "dataset": "nist20",
     "test_split": "split_1"},

    {"folder": "results/dag_canopus_train_public/split_1_rnd1/", 
     "dataset": "canopus_train_public",
     "test_split": "split_1"},

    {"folder": "results/dag_canopus_train_public/split_1_rnd2/", 
     "dataset": "canopus_train_public",
     "test_split": "split_1"},

    {"folder": "results/dag_canopus_train_public/split_1_rnd3/", 
     "dataset": "canopus_train_public",
     "test_split": "split_1"},
]

if debug:
    max_nodes = max_nodes[:3]

for res_entry in res_entries:
    res_folder = Path(res_entry['folder'])
    dataset = res_entry['dataset']
    models = sorted(list(res_folder.rglob("version_0/*.ckpt")))
    split = res_entry['test_split']
    for model in models:
        save_dir_base = model.parent.parent

        save_dir = save_dir_base / "inten_thresh_sweep"
        save_dir.mkdir(exist_ok=True)

        print(f"Saving inten sweep to: {save_dir}")

        pred_dir_folders = []
        form_dir_folders = []
        for max_node in max_nodes:
            save_dir_temp = save_dir / str(max_node)
            save_dir_temp.mkdir(exist_ok=True)

            cmd = f"""python {python_file} \\
            --batch-size {workers} \\
            --dataset-name  {dataset} \\
            --split-name {split}.tsv \\
            --subset-datasets test_only  \\
            --checkpoint {model} \\
            --save-dir {save_dir_temp} \\
            --threshold 0  \\
            --max-nodes {max_node} \\
            """

            pred_dir_folders.append(save_dir_temp)
            device_str = f"CUDA_VISIBLE_DEVICES={devices}"
            cmd = f"{device_str} {cmd}"
            print(cmd + "\n")
            subprocess.run(cmd, shell=True)

            # Convert to form files from dag
        for pred_dir in pred_dir_folders:
            tree_pred_folder = pred_dir / "tree_preds"
            form_pred_folder = pred_dir / "form_preds"
            form_pred_folder.mkdir(exist_ok=True)
            cmd = f"""python data_scripts/dag/dag_to_subform.py \\
                --num-workers 0 \\
                --dag-folder {tree_pred_folder} \\
                --out-dir {form_pred_folder} \\
                --all-h-shifts
            """
            subprocess.run(cmd, shell=True)
            form_dir_folders.append(form_pred_folder)

        res_files = []
        for pred_dir in form_dir_folders:
            analysis_cmd = f"""python analysis/form_pred_eval.py \\
                --dataset {dataset} \\
                --tree-pred-folder {pred_dir} \\
                --subform-name {subform_name}
            """
            res_files.append(pred_dir.parent / "pred_eval.yaml")
            print(analysis_cmd + "\n")
            subprocess.run(analysis_cmd, shell=True)

        ## Run cleanup now
        new_entries = []
        for res_file in res_files:
            new_data = yaml.safe_load(open(res_file, "r"))
            thresh = res_file.parent.stem
            new_entry = {"nm_nodes": thresh}
            new_entry.update({k: v for k, v in new_data.items() if "avg" in k})
            new_entries.append(new_entry)

        df = pd.DataFrame(new_entries)
        df.to_csv(save_dir / "summary.tsv", sep="\t", index=None)
