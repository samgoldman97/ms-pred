from pathlib import Path
import subprocess


dataset = "canopus_train_public"
dataset = "nist20"
res_folder = Path(f"results/dag_nist_hyperopt/")
res_folder = Path(f"results/dag_{dataset}/")
true_dag_folder = f"data/spec_datasets/{dataset}/subformulae/no_subform/"

splits = ["hyperopt"]
nums = [100]

splits = ["split_1"]
splits = ["scaffold_1"]
nums = [100]

for split in splits:
    for num in nums:
        save_dir = res_folder / split
        save_dir = save_dir / f"preds_train_{num}/"
        train_dir = save_dir.parent / f"preds_train_{num}_inten"
        train_dir.mkdir(exist_ok=True)
        cmd = f"""python data_scripts/dag/add_dag_intens.py \\
                  --pred-dag-folder  {save_dir / 'tree_preds'} \\
                  --true-dag-folder {true_dag_folder} \\
                  --out-dag-folder {train_dir}  \\
                  --num-workers 16 \\
                  --add-raw \\
              """
        print(cmd + "\n")
        subprocess.run(cmd, shell=True)
