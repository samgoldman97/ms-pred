"""predict_smis.py

Make both dag and intensity predictions jointly and revert to binned

"""

import logging
import json
from collections import defaultdict
from datetime import datetime
import yaml
import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl

import ms_pred.common as common
from ms_pred.dag_pred import inten_model, gen_model, joint_model


from rdkit import rdBase
from rdkit import RDLogger

rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.*")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument(
        "--gen-checkpoint",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument(
        "--inten-checkpoint",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument("--threshold", default=0.0, action="store", type=float)
    parser.add_argument("--max-nodes", default=100, action="store", type=int)
    parser.add_argument("--upper-limit", default=1500, action="store", type=int)
    parser.add_argument("--num-bins", default=15000, action="store", type=int)
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    debug = kwargs["debug"]
    common.setup_logger(save_dir, log_name="joint_pred.log", debug=debug)
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    data_dir = Path("")
    if kwargs.get("dataset_name") is not None:
        dataset_name = kwargs["dataset_name"]
        data_dir = Path("data/spec_datasets") / dataset_name

    labels = data_dir / kwargs["dataset_labels"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if kwargs["debug"]:
        df = df[:10]

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "debug_special":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
            names = names[:5]
            names = ["CCMSLIB00000001590"]
            kwargs["debug"] = True
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    # Create model and load
    # Load from checkpoint
    gen_checkpoint = kwargs["gen_checkpoint"]
    inten_checkpoint = kwargs["inten_checkpoint"]

    torch.set_num_threads(1)

    inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_checkpoint)
    gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_checkpoint)

    # Build joint model class

    logging.info(
        f"Loaded gen / inten models from {gen_checkpoint} & {inten_checkpoint}"
    )

    model = joint_model.JointModel(
        gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj
    )

    # Don't use GPU for parallel
    assert not kwargs["gpu"]

    with torch.no_grad():
        model.eval()
        model.freeze()
        gpu = kwargs["gpu"]
        device = "cuda" if gpu else "cpu"
        model.to(device)

        binned_out = kwargs["binned_out"]

        def single_predict_mol(entry):
            torch.set_num_threads(1)
            smi = entry["smiles"]
            adduct = entry["ionization"]
            full_output = model.predict_mol(
                smi,
                adduct=adduct,
                threshold=kwargs["threshold"],
                device=device,
                max_nodes=kwargs["max_nodes"],
                binned_out=binned_out,
            )
            if binned_out:
                # Get only index
                output_spec = full_output["spec"][0]
                best_inds = None

                if kwargs["sparse_out"]:
                    sparse_k = kwargs["sparse_k"]
                    best_inds = np.argsort(output_spec, -1)[::-1][:sparse_k]
                    best_intens = np.take_along_axis(output_spec, best_inds, -1)
                    output_spec = np.stack([best_inds, best_intens], -1)

                update_dict = {}
                for param_k, param_v in full_output.items():
                    if param_k in ["spec", "forms", "masses"]:
                        continue

                    # Shrink it to only top k, ordering inds, intens
                    if kwargs["sparse_out"]:
                        best_params = np.take_along_axis(param_v, best_inds, -1)
                        param_v = np.stack([best_inds, best_params], -1)

                    update_dict[param_k] = param_v

                out = {"preds": output_spec}
                out.update(update_dict)
            else:
                out = full_output
            return out

        all_rows = [j for _, j in df.iterrows()]
        preds = []
        if kwargs["debug"]:
            for j in all_rows:
                pred = single_predict_mol(j)
                preds.append(pred)
        else:
            preds = common.chunked_parallel(
                all_rows, single_predict_mol, max_cpu=kwargs["batch_size"], spawn=True
            )

        # Export out
        if binned_out:
            output_keys = set(preds[0].keys())
            update_dict = {}
            for k in output_keys:
                temp = [i[k] for i in preds]
                update_dict[k] = np.stack(temp, 0)

            spec_names_ar = [i["spec"] for i in all_rows]
            smiles_ar = np.array([i["smiles"] for i in all_rows])
            inchikeys = [common.inchikey_from_smiles(i) for i in smiles_ar]

            output = {
                "smiles": smiles_ar,
                "ikeys": inchikeys,
                "spec_names": spec_names_ar,
                "num_bins": 15000,
                "upper_limit": 1500,
                "sparse_out": kwargs["sparse_out"],
            }
            output.update(update_dict)

            out_file = Path(kwargs["save_dir"]) / "binned_preds.p"
            with open(out_file, "wb") as fp:
                pickle.dump(output, fp)
        else:
            for pred_obj, row in zip(preds, all_rows):
                spec_name = row["spec"]
                ionization = row["ionization"]
                pred_obj["name"] = spec_name
                pred_obj["adduct"] = ionization
                save_path = Path(kwargs["save_dir"]) / "tree_preds_inten"
                save_path.mkdir(exist_ok=True)
                out_file = save_path / f"pred_{spec_name}.json"
                with open(out_file, "w") as fp:
                    json.dump(pred_obj, fp, indent=2)


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
