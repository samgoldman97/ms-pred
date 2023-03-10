from pathlib import Path
import subprocess

dataset_name = "nist20"
dataset_name = "canopus_train_public"

res_folder = Path(f"results/gnn_baseline_{dataset_name}")
python_file = "src/ms_pred/gnn_pred/predict.py"
devices = ",".join(["0"])

for model in res_folder.rglob("version_0/*.ckpt"):
    save_dir = model.parent.parent
    split = save_dir.name
    save_dir = save_dir / "preds"
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset_name} \\
    --split-name {split}.tsv \\
    --subset-datasets test_only  \\
     --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu"""
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    out_binned = save_dir / "binned_preds.p"
    eval_cmd = f"""python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform \\
    --dataset {dataset_name}  \\
    """
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True)
