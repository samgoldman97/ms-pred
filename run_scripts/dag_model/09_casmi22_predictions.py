""" Export model predictions for casmi22 after processing data via notebook"""
import yaml
from pathlib import Path
import subprocess

pred_file = "src/ms_pred/dag_pred/predict_smis.py"
devices = ",".join(["1"])
subform_name = "no_subform"
max_nodes = 100
dist = "cos"
test_entries = [
    {
        "test_dataset": "casmi22",
        "dataset": "canopus_train_public",
        "split": "all_split",
        "binned_out": True,
        "folder": "split_1_rnd1"
    },
]

for test_entry in test_entries:
    binned_out = test_entry['binned_out']
    dataset = test_entry['dataset']
    test_dataset = test_entry['test_dataset']
    split = test_entry['split']
    folder = test_entry['folder']
    inten_model = Path(f"results/dag_inten_{dataset}/{folder}/version_0/best.ckpt")
    binned_out_flag = "--binned-out" if binned_out else ""

    save_dir = inten_model.parent.parent
    save_dir = save_dir / f"preds_export_{test_dataset}"

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    save_dir.mkdir(exist_ok=True, parents=True)

    labels = f"labels.tsv"
    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --batch-size 32 \\
    --dataset-name {test_dataset} \\
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
    #--sparse-out \\
    #--sparse-k 100 \\
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    out_binned = save_dir / "binned_preds.p"
    eval_cmd = f"""
    python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform \\
    --dataset {test_dataset}
    """
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True)
