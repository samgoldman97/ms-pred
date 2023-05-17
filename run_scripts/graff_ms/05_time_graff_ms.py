import json
import pandas as pd
import time
from pathlib import Path
import subprocess

dataset = "canopus_train_public"
dataset = "nist20"
dataset_labels = "timer_labels.tsv"
labels = Path(f"data/spec_datasets/{dataset}") / dataset_labels
num_mols = len(pd.read_csv(labels, sep="\t"))


res_folder = Path(f"results/graff_ms_baseline_{dataset}")
python_file = "src/ms_pred/graff_ms/predict.py"
devices = ",".join(["3"])

for model in res_folder.rglob("version_0/*.ckpt"):
    save_dir = model.parent.parent
    time_res = save_dir / "time_out.json"
    split = save_dir.name
    save_dir = save_dir / "preds_time"
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {python_file} \\
    --batch-size 1 \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --subset-datasets none \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --num-workers 0 \\
    --dataset-labels {dataset_labels}
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    start = time.time()
    subprocess.run(cmd, shell=True)
    end = time.time()
    seconds = end - start
    out_dict = {"time (s)": seconds, "mols": num_mols}
    print(f"Time taken for preds {seconds}")
    json.dump(out_dict, open(time_res, "w"))
