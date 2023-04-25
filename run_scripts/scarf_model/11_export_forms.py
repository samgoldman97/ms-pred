import yaml
from pathlib import Path
import subprocess

pred_file = "src/ms_pred/scarf_pred/predict_smis.py"
devices = ",".join(["3"])
subform_name = "no_subform"
max_nodes = 300
dataset = "nist20"
dist = "cos"
binned_out = False

binned_out_flag = "--binned-out" if binned_out else ""

inten_dir = Path(f"results/scarf_inten_{dataset}")  # _{max_nodes}")

for inten_model in inten_dir.rglob("version_0/*.ckpt"):
    save_dir = inten_model.parent.parent / f"preds_export_{dataset}"
    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["formula_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    split = save_dir.parent.name
    save_dir.mkdir(exist_ok=True)

    # split = "split_nist"

    labels = f"labels.tsv"
    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size 32 \\
    --dataset-name {dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {max_nodes} \\
    --split-name {split}.tsv   \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    {binned_out_flag} \\
    --num-workers 16 \\
    --subset-datasets test_only \\
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
