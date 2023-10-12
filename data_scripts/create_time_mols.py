from pathlib import Path
import numpy as np
import pandas as pd

dataset = "nist20"
data_path = Path(f"data/spec_datasets/{dataset}")
labels = data_path / "labels.tsv"
out = data_path.parent / "sample_labels.tsv"
num_sample = 100

df = pd.read_csv(labels, sep="\t")
examples = len(df)
sampled_inds = np.random.choice(examples, num_sample, replace=False)
df = df.iloc[sampled_inds].reset_index(drop=True)
df.to_csv(out, index=None, sep="\t")
