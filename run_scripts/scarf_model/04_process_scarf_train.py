from pathlib import Path
import subprocess


dataset = "canopus_train_public"
dataset = "nist20"
res_folder = Path("results/scarf_nist_hyperopt")
res_folder = Path(f"results/scarf_{dataset}")
true_dag_folder = f"data/spec_datasets/{dataset}/subformulae/no_subform/"
splits = ["split_1", "split_2", "split_3"]
splits = ["split_1"]
nums = [100]

for split in splits:
    for num in nums:
        save_dir = res_folder / split
        save_dir = save_dir / f"preds_train_{num}/"
        train_dir = save_dir.parent / f"preds_train_{num}_inten"
        train_dir.mkdir(exist_ok=True)
        cmd = f"""python data_scripts/forms/03_add_form_intens.py --pred-form-folder 
               {save_dir / 'form_preds'} --true-form-folder {true_dag_folder}
              --out-form-folder {train_dir}  --num-workers 16 --add-raw --binned-add
              """
        cmd = cmd.replace("\n", " ")
        print(cmd + "\n")
        subprocess.run(cmd, shell=True)
