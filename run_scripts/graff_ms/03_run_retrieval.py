from pathlib import Path
import subprocess

dataset = "canopus_train_public"
dataset = "nist20"
res_folder = Path(f"results/graff_ms_baseline_{dataset}_with_zero/")
res_folder = Path(f"results/graff_ms_baseline_{dataset}/")
res_folder = Path(f"results/graff_ms_baseline_{dataset}_with_zero/")
pred_file = "src/ms_pred/graff_ms/predict.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_binned.py"
devices = ",".join(["0"])
subform_name = "no_subform"

num_workers = 32

valid_splits = ["split_1"]
split_override = "split_1"
maxk = 50  # None

for model in res_folder.rglob("version_0/*.ckpt"):
    split = model.parent.parent.name
    if split not in valid_splits:
        continue

    if split_override is not None:
        split = split_override

    save_dir = model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)

    labels = f"retrieval/cands_df_{split}_{maxk}.tsv"
    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size 32  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --num-workers {num_workers} \\
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
    --binned-pred-file {save_dir / 'binned_preds.p'} \\
    --dist-fn cos \\
        """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)