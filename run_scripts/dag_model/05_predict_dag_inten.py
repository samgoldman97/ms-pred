from pathlib import Path
import subprocess
import argparse

python_file = "src/ms_pred/dag_pred/predict_inten.py"
node_num = 100
num_workers = 32
test_entries = [
    {"dataset": "nist20", "split": "scaffold_1", "folder": "scaffold_1"},
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd1"},
    {"dataset": "canopus_train_public", "split": "split_1", "folder": "split_1_rnd1"},
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd2"},
    {"dataset": "canopus_train_public", "split": "split_1", "folder": "split_1_rnd2"},
    {"dataset": "nist20", "split": "split_1", "folder": "split_1_rnd3"},
    {"dataset": "canopus_train_public", "split": "split_1", "folder": "split_1_rnd3"},
]
devices = ",".join(["1"])

for test_entry in test_entries:
    split = test_entry['split']
    dataset = test_entry['dataset']
    folder = test_entry['folder']

    base_formula_folder = Path(f"results/dag_{dataset}")
    res_folder = Path(f"results/dag_inten_{dataset}/")
    model = res_folder / folder / "version_0/best.ckpt"

    if not model.exists(): 
        continue

    save_dir = model.parent.parent

    save_dir = save_dir / "preds"

    # Note: Must use preds_train_01
    magma_dag_folder = (
        base_formula_folder / folder / f"preds_train_{node_num}/tree_preds"
    )
    print(magma_dag_folder)
    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu \\
    --num-workers 0 \\
    --magma-dag-folder {magma_dag_folder} \\
    --subset-datasets test_only \\
    --binned-out
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Eval it
    out_binned = save_dir / "binned_preds.p"
    eval_cmd = f"""
    python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform \\
    --dataset {dataset}  \\
    """
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True)
