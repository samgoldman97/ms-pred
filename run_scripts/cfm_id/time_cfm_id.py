import pandas as pd
import subprocess
from pathlib import Path
from ms_pred import common
import json
import time


#dataset = "nist20"
labels = f"data/spec_datasets/timer_labels.tsv"
res_folder = Path(f"results/cfm_id_timer/")
time_res = res_folder / "time_out.json"

res_folder.mkdir(exist_ok=True)

cfm_inputs = []
cfm_output_specs = res_folder / "cfm_out"

df = pd.read_csv(labels, sep="\t")
input_items = [f"{i} {j}" for i, j in df[["spec", "smiles"]].values]
input_str = "\n".join(input_items)
input_file = res_folder / "input.txt"
open(input_file, "w").write(input_str)


cfm_command = f"""cfm-predict '/cfmid/public/{input_file}' 0.001 \\
    /trained_models_cfmid4.0/[M+H]+/param_output.log \\
    /trained_models_cfmid4.0/[M+H]+/param_config.txt 1 \\
    /cfmid/public/{cfm_output_specs}"""


strt = time.time()
docker_str = f""" docker run --rm=true -v $(pwd):/cfmid/public/ \\
                  -i wishartlab/cfmid:latest  \\
                  sh -c  "{cfm_command}"
              """
print(docker_str)
subprocess.run(docker_str, shell=True)
end = time.time()
seconds = end - strt
num_items = len(input_items)
out_dict = {"time (s)": seconds, "mols": num_items}
print(f"Time taken for preds {seconds}")
json.dump(out_dict, open(time_res, "w"))
