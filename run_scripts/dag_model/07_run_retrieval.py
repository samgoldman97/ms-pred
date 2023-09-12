import yaml
from pathlib import Path
import subprocess

pred_file = "src/ms_pred/dag_pred/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_binned.py"
devices = ",".join(["3"])
subform_name = "no_subform"
max_nodes = 100
dataset = "nist20"
dataset = "canopus_train_public"
dist = "cos"
split_override = None
maxk = None

inten_dir = Path(f"results/dag_inten_{dataset}")  # _{max_nodes}")
valid_splits = ["split_1"]

for inten_model in inten_dir.rglob("version_0/*.ckpt"):
    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    split = inten_model.parent.parent.name

    if split not in valid_splits:
        continue

    save_dir = inten_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)
    if split_override is not None:
        split = split_override

    labels = f"retrieval/cands_df_{split}_{maxk}.tsv"
    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size 32  \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {max_nodes} \\
    --split-name {split}.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --binned-out \\
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name} \\
    --binned-pred-file {save_dir / 'binned_preds.p'} \\
    --dist-fn {dist} \\
    """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval random baseline
    cmd = f"""python {retrieve_file} \\
    --dataset {dataset} \\
    --formula-dir-name {subform_name} \\
    --binned-pred-file {save_dir / 'binned_preds.p'} \\
    --dist-fn random \\
    """

    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
