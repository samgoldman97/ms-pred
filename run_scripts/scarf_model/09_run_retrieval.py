import yaml
from pathlib import Path
import subprocess

num_workers = 64
pred_file = "src/ms_pred/scarf_pred/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/retrieval_binned.py"
devices = ",".join(["3"])
subform_name = "no_subform"
max_nodes = 100
max_nodes = 300

dataset = "canopus_train_public"
dataset = "nist20"
dist = "cos"

inten_dir = Path(f"results/scarf_inten_{dataset}_100")  # _{max_nodes}")
inten_dir = Path(f"results/scarf_inten_{dataset}")  # _{max_nodes}")

valid_splits = ["split_1"]
split_override = "split_1_500" 
maxk=None

for inten_model in inten_dir.rglob("version_0/*.ckpt"):
    split = inten_model.parent.parent.name
    if split not in valid_splits:
        continue

    if split_override is not None:
        split = split_override

    save_dir = inten_model.parent.parent / f"retrieval_{dataset}_{split}_{maxk}"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["formula_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

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
