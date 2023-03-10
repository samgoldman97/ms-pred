from pathlib import Path
import subprocess

dataset = "nist20"
dataset = "canopus_train_public"
#res_folder = Path(f"results/ffn_baseline_epoch_ablation_canopus_train_public")
res_folder = Path(f"results/ffn_baseline_{dataset}/")

pred_file = "src/ms_pred/ffn_pred/predict.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_binned.py"
devices = ",".join(["3"])
subform_name = "no_subform"
dist = "cos"
split_override = "split_1"
split_override = None


for model in res_folder.rglob("version_0/*.ckpt"):
    save_dir = model.parent.parent / f"retrieval_{dataset}"
    split = save_dir.parent.name
    if split_override is not None:
        split = split_override
    save_dir.mkdir(exist_ok=True)

    labels = f"retrieval/cands_df_{split}.tsv"
    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size 32  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --split-name {split}.tsv   \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --gpu"""
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name} \\
    --binned-pred-file {save_dir / 'fp_preds.p'} \\
    --dist-fn {dist} \\
    """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
