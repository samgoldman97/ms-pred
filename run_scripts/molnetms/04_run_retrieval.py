from pathlib import Path
import subprocess


dataset = "canopus_train_public"
dataset = "nist20"
res_folder = Path(f"results/molnetms_baseline_{dataset}/")
pred_file = "src/ms_pred/molnetms/predict.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_binned.py"
devices = ",".join(["1"])
subform_name = "no_subform"


for model in res_folder.rglob("version_0/*.ckpt"):
    save_dir = model.parent.parent / f"retrieval_{dataset}"
    split = save_dir.parent.name
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
    --binned-pred-file {save_dir / 'binned_preds.p'} \\
    --dist-fn cos \\
        """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
