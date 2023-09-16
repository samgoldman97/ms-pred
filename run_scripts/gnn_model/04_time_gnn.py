import json
import pandas as pd
import time
from pathlib import Path
import subprocess

python_file = "src/ms_pred/gnn_pred/predict.py"

test_entries = [{"dataset": "nist20",
                 "labels": "data/spec_datasets/sample_labels.tsv" 
                 "train_split": "split_1"
                 }]
devices = ",".join(["3"])

for test_entry in test_entries:
    dataset = test_entry['dataset']
    labels = test_entry['labels']
    train_split = test_entry['train_split']
    res_folder = Path(f"results/gnn_baseline_{dataset}/")
    model = res_folder / train_split / f"version_0/best.ckpt"
    num_mols = len(pd.read_csv(labels, sep="\t"))

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
