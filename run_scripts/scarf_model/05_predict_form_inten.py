from pathlib import Path
import subprocess
import argparse

python_file = "src/ms_pred/scarf_pred/predict_inten.py"
devices = ",".join(["3"])
node_num = 300

test_entries = [
    {"dataset": "canopus_train_public", "split": "split_1", },
    {"dataset": "nist20", "split": "split_1", },
    {"dataset": "nist20", "split": "scaffold_1", }
]

for test_entry in test_entries:
    dataset = test_entry['dataset']
    split = test_entry['split']
    res_folder = Path(f"results/scarf_inten_{dataset}/")
    model =  res_folder / split / f"version_0/best.ckpt"
    base_formula_folder = Path(f"results/scarf_{dataset}")
    save_dir = model.parent.parent
    split = save_dir.name

    save_dir = save_dir / "preds"

    # Note: Must use preds_train_01
    formula_folder = base_formula_folder / split / f"preds_train_{node_num}/form_preds"
    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu \\
    --num-workers 0 \\
    --subset-datasets test_only \\
    --formula-folder {formula_folder} \\
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
