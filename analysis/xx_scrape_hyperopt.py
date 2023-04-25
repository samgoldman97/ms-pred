""" xx_scrape_hyperopt.

If hyperopt is interrupted, pull results from the folder and format

"""
import json
from pathlib import Path
import pandas as pd
import numpy as np


res_folder = Path("results/dag_gen_hyperopt/score_function_2023-01-01_23-55-51")
res_folder = Path("results/dag_inten_hyperopt_v2/score_function_2023-01-05_21-25-26/")


all_res = []
for res in res_folder.rglob("result.json"):
    try:
        lines = list(open(res, "r").readlines())
        res_dicts = [json.loads(i) for i in lines]
        res_dict = min(res_dicts, key=lambda x: x["val_loss"])
        min_loss = res_dict["val_loss"]
        params = json.load(open(res.parent / "params.json", "r"))
        params["min_loss"] = min_loss
        all_res.append(params)
    except:
        pass
df = pd.DataFrame(all_res)
df = df.sort_values(by="min_loss", axis=0).reset_index(drop=True)
df.to_csv(res_folder.parent / "sorted_res.tsv", index=None, sep="\t")
row_0 = df.iloc[0]
row_0 = dict({i: str(k) for i, k in row_0.items()})
temp = json.dumps(row_0, indent=2)
with open(res_folder.parent / "best_params.json", "w") as fp:
    fp.write(temp)

print(df)
print(temp)
