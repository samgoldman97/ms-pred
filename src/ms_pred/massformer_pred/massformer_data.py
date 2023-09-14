import logging
import json
import numpy as np

import torch
from rdkit import Chem
from torch.utils.data.dataset import Dataset
import dgl

import ms_pred.common as common
from ._massformer_graph_featurizer import MassformerGraphFeaturizer


class BinnedDataset(Dataset):
    """BinnedDataset."""

    def __init__(
        self,
        df,
        data_dir,
        num_bins,
        num_workers=0,
        upper_limit=1500,
        form_dir_name: str = "subform_20",
        use_ray=False,
        **kwargs,
    ):
        self.df = df
        self.num_bins = num_bins
        self.num_workers = num_workers
        self.upper_limit = upper_limit
        self.bins = np.linspace(0, self.upper_limit, self.num_bins)
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        self.graph_featurizer = MassformerGraphFeaturizer()

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.weights = [common.ExactMolWt(i) for i in self.mols]
            self.mol_graphs = [self.graph_featurizer(el) for el in self.mols]
        else:
            mol_from_smi = lambda x: Chem.MolFromSmiles(x)
            self.mols = common.chunked_parallel(
                self.smiles,
                mol_from_smi,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=600,
                max_retries=3,
                use_ray=use_ray,
            )
            self.weights = common.chunked_parallel(
                self.mols,
                lambda x: common.ExactMolWt(x),
                chunks=100,
                max_cpu=self.num_workers,
                timeout=600,
                max_retries=3,
                use_ray=use_ray,
            )
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.graph_featurizer,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        self.weights = np.array(self.weights)

        # Read in all specs
        self.spec_names = self.df["spec"].values
        spec_files = [
            (data_dir / "subformulae" / f"{form_dir_name}" / f"{spec_name}.json")
            for spec_name in self.spec_names
        ]
        process_spec_file = lambda x: common.bin_form_file(
            x, num_bins=num_bins, upper_limit=upper_limit
        )
        if self.num_workers == 0:
            spec_outputs = [process_spec_file(i) for i in spec_files]
        else:
            spec_outputs = common.chunked_parallel(
                spec_files,
                process_spec_file,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        self.metas, self.spec_ars = zip(*spec_outputs)
        mask = np.array([i is not None for i in self.spec_ars])
        logging.info(f"Could not find tables for {np.sum(~mask)} spec")

        # Self.weights, self. mol_graphs
        self.metas = np.array(self.metas)[mask].tolist()
        self.spec_ars = np.array(self.spec_ars, dtype=object)[mask].tolist()
        self.df = self.df[mask]
        self.spec_names = np.array(self.spec_names)[mask].tolist()
        self.weights = np.array(self.weights)[mask].tolist()
        self.mol_graphs = [
            el for el, valid in zip(self.mol_graphs, mask.tolist()) if valid
        ]

        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        meta = self.metas[idx]
        ar = self.spec_ars[idx]
        graph = self.mol_graphs[idx]
        full_weight = self.weights[idx]
        adduct = self.adducts[idx]
        outdict = {
            "name": name,
            "binned": ar,
            "full_weight": full_weight,
            "adduct": adduct,
            "gf_v2_data": graph,
            "_meta": meta,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return BinnedDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        spec_ars = [j["binned"] for j in input_list]
        graphs = MassformerGraphFeaturizer.collate_func(
            [j["gf_v2_data"] for j in input_list]
        )
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        # Now pad everything else to the max channel dim
        spectra_tensors = torch.stack([torch.tensor(spectra) for spectra in spec_ars])
        full_weight = torch.FloatTensor(full_weight)

        # frag_batch.set_n_initializer(dgl.init.zero_initializer)
        # frag_batch.set_e_initializer(dgl.init.zero_initializer)

        adducts = torch.FloatTensor(adducts)

        return_dict = {
            "spectra": spectra_tensors,
            "gf_v2_data": graphs,
            "names": names,
            "adducts": adducts,
            "full_weight": full_weight,
        }
        return return_dict


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(self, df, num_workers: int = 0, **kwargs):

        self.df = df
        self.num_workers = num_workers
        self.graph_featurizer = MassformerGraphFeaturizer()
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        self.spec_names = ["" for i in self.smiles]
        if "spec" in list(self.df.keys()):
            self.spec_names = self.df["spec"].values

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.weights = [common.ExactMolWt(i) for i in self.mols]
            self.mol_graphs = [self.graph_featurizer(el) for el in self.mols]
        else:
            self.mols = common.chunked_parallel(
                self.smiles,
                Chem.MolFromSmiles,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
            self.weights = common.chunked_parallel(
                self.mols,
                common.ExactMolWt,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.graph_featurizer,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )

        # Extract
        self.weights = np.array(self.weights)
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        graph = self.mol_graphs[idx]
        full_weight = self.weights[idx]
        spec_name = self.spec_names[idx]
        adduct = self.adducts[idx]
        outdict = {
            "smi": smi,
            "gf_v2_data": graph,
            "adduct": adduct,
            "full_weight": full_weight,
            "spec_name": spec_name,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["smi"] for j in input_list]
        spec_names = [j["spec_name"] for j in input_list]
        graphs = MassformerGraphFeaturizer.collate_func(
            [j["gf_v2_data"] for j in input_list]
        )
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)

        full_weight = torch.FloatTensor(full_weight)
        return_dict = {
            "gf_v2_data": graphs,
            "spec_names": spec_names,
            "names": names,
            "full_weight": full_weight,
            "adducts": adducts,
        }
        return return_dict
