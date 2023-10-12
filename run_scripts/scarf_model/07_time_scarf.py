import yaml
import json
import pandas as pd
import time
from pathlib import Path
import subprocess

python_file = "src/ms_pred/scarf_pred/predict_smis.py"
devices = ",".join(["3"])
node_num = 300

test_entries = [
    {"dataset" : "nist20", "labels": "data/spec_datasets/sample_labels.tsv",
    "train_split": "split_1_rnd1"}
]

for test_entry in test_entries:
    dataset = test_entry['dataset']
    labels = test_entry['labels']
    train_split = test_entry['train_split']

    res_folder = Path(f"results/scarf_inten_{dataset}/")
    base_formula_folder = Path(f"results/scarf_{dataset}")
    model  = res_folder / train_split / f"version_0/best.ckpt"
    num_mols = len(pd.read_csv(labels, sep="\t"))

    save_dir = model.parent.parent
    time_res = save_dir / "time_out.json"
    save_dir = save_dir / "preds_time"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["formula_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    # Note: Must use preds_train_01
    cmd = f"""python {python_file} \\
    --batch-size 1 \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {model} \\
    --max-nodes {node_num} \\
    --sparse-out \\
    --sparse-k 100 \\
    --save-dir {save_dir} \\
    --num-workers 0 \\
    --subset-datasets none \\
    --dataset-labels  {labels} \\
    --binned-out
    #--gpu \\
    """
    print(cmd + "\n")
    start = time.time()
    subprocess.run(cmd, shell=True)
    end = time.time()
    seconds = end - start
    out_dict = {"time (s)": seconds, "mols": num_mols}
    print(f"Time taken for preds {seconds}")
    json.dump(out_dict, open(time_res, "w"))
