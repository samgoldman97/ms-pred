import logging
from pathlib import Path
from typing import List
from functools import partial
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

import torch
from torch.utils.data.dataset import Dataset
import dgl

import ms_pred.common as common
import ms_pred.massformer_pred._massformer_graph_featurizer as mformer
from torch_geometric.data.data import Data as pyg_data


def process_form_file(
    form_dict_file,
):
    """process_form_file."""
    with open(form_dict_file, "r") as fp:
        form_dict = json.load(fp)

    root_form = form_dict["cand_form"]
    out_tbl = form_dict["output_tbl"]

    if out_tbl is None:
        return None

    intens = out_tbl["ms2_inten"]
    formulae = out_tbl["formula"]
    raw_spec = out_tbl.get("raw_spec")

    out_dict = dict(
        root_form=root_form, intens=intens, formulae=formulae, raw_spec=raw_spec
    )
    return out_dict


class AutoregrDataset(Dataset):
    """AutoregrDataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        graph_featurizer,
        file_map: dict,
        num_workers=0,
        root_embedder: str = "gnn",
        use_ray: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            df:
            data_dir:
            graph_featurizer
            file_map (dict): file_map
            num_workers:
            use_ray (bool): use_ray
            kwargs:
        """
        self.df = df
        self.num_workers = num_workers
        self.file_map = file_map
        self.max_atom_ct = common.MAX_ATOM_CT
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        valid_specs = [i in self.file_map for i in self.df["spec"].values]
        self.df_sub = self.df[valid_specs]
        if len(self.df_sub) == 0:
            self.spec_names = []
            self.name_to_dict = {}
        else:
            self.spec_names = self.df_sub["spec"].values
            self.name_to_dict = self.df_sub.set_index(
                "spec").to_dict(orient="index")

        for i in self.name_to_dict:
            self.name_to_dict[i]["formula_file"] = self.file_map[i]

        # Load smiles
        self.smiles = self.df_sub["smiles"].values
        self.graph_featurizer = graph_featurizer
        self.num_atom_feats = self.graph_featurizer.num_atom_feats
        self.num_bond_feats = self.graph_featurizer.num_bond_feats

        self.root_embedder = root_embedder
        self.root_encode_fn = None
        if root_embedder == "gnn":
            self.root_encode_fn = self.graph_featurizer.get_dgl_graph
        elif root_embedder == "fp":
            self.root_encode_fn = common.get_morgan_fp
        elif root_embedder == "graphormer":
            self.root_encode_fn = mformer.MassformerGraphFeaturizer()
        else:
            raise ValueError()

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.mol_graphs = [self.root_encode_fn(i) for i in self.mols]
        else:
            def mol_from_smi(x): return Chem.MolFromSmiles(x)
            self.mols = common.chunked_parallel(
                self.smiles,
                mol_from_smi,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=600,
                max_retries=3,
                use_ray=use_ray,
            )
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.root_encode_fn,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        logging.info(
            f"{len(self.df_sub)} of {len(self.df)} spec have form dicts.")
        self.form_files = [
            self.name_to_dict[i]["formula_file"] for i in self.spec_names
        ]

        # Read in all trees necessary in parallel
        # Process trees jointly; sacrifice mmeory for speed
        process_fn = partial(
            process_form_file,
        )
        if self.num_workers > 0:
            self.spec_forms = common.chunked_parallel(
                self.form_files,
                process_fn,
                max_cpu=self.num_workers,
                chunks=50,
                timeout=1200,
                use_ray=use_ray,
            )
        else:
            self.spec_forms = [process_fn(i) for i in tqdm(self.form_files)]

        self.name_to_smiles = dict(zip(self.spec_names, self.smiles))
        self.name_to_mols = dict(zip(self.spec_names, self.mol_graphs))
        self.name_to_forms = dict(zip(self.spec_names, self.spec_forms))
        self.spec_names = [
            i for i in self.spec_names if self.name_to_forms[i] is not None
        ]

        self.name_to_adducts = {
            i: common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        }

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):

        name = self.spec_names[idx]

        smiles = self.name_to_smiles[name]
        mol_graph = self.name_to_mols[name]
        spec_form_obj = self.name_to_forms[name]
        adduct = self.name_to_adducts[name]

        full_formula = spec_form_obj["root_form"]
        form_list = spec_form_obj["formulae"]
        form_intens = spec_form_obj["intens"]

        # Get dense vec of all the formulae
        all_form_vecs = [common.formula_to_dense(i) for i in form_list]
        full_vec = common.formula_to_dense(full_formula)

        # Create a data structure mapping to all true options
        # Prev layer options --> shape opts x atom dim
        atom_dim = full_vec.shape[0]
        prev_opts = np.zeros(atom_dim)[None, :]
        nonzero_inds = np.nonzero(full_vec)[0]

        sorted_intens = np.argsort(form_intens)[::-1]
        form_intens = np.array(form_intens)[sorted_intens]
        all_form_vecs = np.array(all_form_vecs)[sorted_intens]
        atom_inds = np.array(
            [nonzero_inds for i in range(all_form_vecs.shape[0])]).reshape(-1)

        all_form_vecs = all_form_vecs[:, nonzero_inds]
        all_form_vecs = all_form_vecs.reshape(-1)
        atom_inds = atom_inds.reshape(-1)

        outdict = {
            # Inputs
            "name": name,
            "smiles": smiles,
            "mol_graph": mol_graph,
            "base_formula": full_vec,
            "adduct": adduct,

            # outputs
            "all_forms": all_form_vecs,
            "atom_inds": atom_inds,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return AutoregrDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        mol_graphs = [j["mol_graph"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)

        if isinstance(mol_graphs[0], dgl.DGLGraph):
            batched_graph = dgl.batch(mol_graphs)
        elif isinstance(mol_graphs[0], np.ndarray):
            batched_graph = torch.FloatTensor(np.vstack(mol_graphs))
        elif isinstance(mol_graphs[0], pyg_data):
            batched_graph = mformer.MassformerGraphFeaturizer.collate_func(
                mol_graphs)
        else:
            raise NotImplementedError()

        formula_ars = [j["base_formula"] for j in input_list]
        formula_tensors = torch.FloatTensor(formula_ars)

        targ_vectors = [torch.FloatTensor(i['all_forms']) for i in input_list]
        ind_vectors = [torch.FloatTensor(i['atom_inds']) for i in input_list]

        targ_lens = torch.FloatTensor([len(i) for i in targ_vectors])

        # Pad and stack targ vectors with torch rnn
        targ_vectors = torch.nn.utils.rnn.pad_sequence(targ_vectors,
                                                       batch_first=True)
        ind_vectors = torch.nn.utils.rnn.pad_sequence(ind_vectors,
                                                      batch_first=True)

        output = {
            "names": names,
            "graphs": batched_graph,
            "formula_tensors": formula_tensors,
            "adducts": adducts,
            "targ_lens": targ_lens,

            # Atom inds
            "atom_inds": ind_vectors,
            "targ_vectors": targ_vectors,
        }
        return output


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(
        self,
        df,
        graph_featurizer,
        num_workers: int = 0,
        root_embedder: str = "gnn",
        **kwargs,
    ):

        self.df = df
        self.num_workers = num_workers
        self.graph_featurizer = graph_featurizer
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        self.spec_names = ["" for i in self.smiles]
        if "spec" in list(self.df.keys()):
            self.spec_names = self.df["spec"].values

        self.root_embedder = root_embedder
        self.root_encode_fn = None
        if root_embedder == "gnn":
            self.root_encode_fn = self.graph_featurizer.get_dgl_graph
        elif root_embedder == "fp":
            self.root_encode_fn = common.get_morgan_fp
        else:
            raise ValueError()

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.base_formulae = [common.form_from_smi(i) for i in self.smiles]
            self.mol_graphs = [self.root_encode_fn(i) for i in self.mols]
        else:
            self.mols = common.chunked_parallel(
                self.smiles,
                lambda x: Chem.MolFromSmiles(x),
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.root_encode_fn,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
            self.base_formulae = common.chunked_parallel(
                self.smiles,
                common.form_from_smi,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        mol_graph = self.mol_graphs[idx]
        spec_name = self.spec_names[idx]
        full_formula = self.base_formulae[idx]
        full_vec = common.formula_to_dense(full_formula)
        adduct = self.adducts[idx]

        # Ideally the batch should have: The formula vector
        outdict = {
            "name": smi,
            "spec_name": spec_name,
            "mol_graph": mol_graph,
            "base_formula": full_vec,
            "adduct": adduct,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        spec_names = [j["spec_name"] for j in input_list]
        mol_graphs = [j["mol_graph"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        adducts = torch.FloatTensor(adducts)

        if isinstance(mol_graphs[0], dgl.DGLGraph):
            batched_graph = dgl.batch(mol_graphs)
        elif isinstance(mol_graphs[0], np.ndarray):
            batched_graph = torch.FloatTensor(np.vstack(mol_graphs))
        else:
            raise NotImplementedError()

        formula_ars = [j["base_formula"] for j in input_list]

        # Expand out all examples
        formula_tensors = torch.stack(
            [torch.tensor(formula_ar) for formula_ar in formula_ars]
        )
        output = {
            "names": names,
            "mol_graphs": batched_graph,
            "formula_tensors": formula_tensors,
            "spec_names": spec_names,
            "adducts": adducts,
        }
        return output
