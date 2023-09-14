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
    with open(str(form_dict_file), "r") as fp:
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


class IntenDataset(Dataset):
    """ScarfDataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        graph_featurizer,
        form_map: dict,
        num_workers=0,
        use_ray: bool = False,
        root_embedder: str = "gnn",
        binned_targs: bool = False,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            df (pd.DataFrame): _description_
            data_dir (Path): _description_
            graph_featurizer (_type_): _description_
            form_map (dict): _description_
            num_workers (int, optional): _description_. Defaults to 0.
            use_ray (bool, optional): _description_. Defaults to False.
            root_embedder (str, optional): _description_. Defaults to "gnn".
            binned_targs (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """

        self.df = df
        self.num_workers = num_workers
        self.file_map = form_map
        self.max_atom_ct = common.MAX_ATOM_CT
        self.binned_targs = binned_targs
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        valid_specs = [i in self.file_map for i in self.df["spec"].values]
        self.df_sub = self.df[valid_specs]
        if len(self.df_sub) == 0:
            self.spec_names = []
            self.name_to_dict = {}
        else:
            self.spec_names = self.df_sub["spec"].values
            self.name_to_dict = self.df_sub.set_index("spec").to_dict(orient="index")

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
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.root_encode_fn,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        logging.info(f"{len(self.df_sub)} of {len(self.df)} spec have form dicts.")
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
            self.spec_forms = [process_fn(i) for i in self.form_files]

        self.name_to_smiles = dict(zip(self.spec_names, self.smiles))
        self.name_to_mols = dict(zip(self.spec_names, self.mol_graphs))
        self.name_to_forms = dict(zip(self.spec_names, self.spec_forms))
        len_spec_names = len(self.spec_names)
        self.spec_names = [
            i
            for i in self.spec_names
            if self.name_to_forms.get(i) is not None
            and len(self.name_to_forms.get(i)["formulae"]) > 0
        ]
        post_len_spec_names = len(self.spec_names)
        logging.info(f"{post_len_spec_names} of {len_spec_names} have nonzero intens.")
        self.spec_names = np.array(self.spec_names)
        self.name_to_root_form = {
            i: self.name_to_forms[i]["root_form"] for i in self.spec_names
        }

        self.bins = np.linspace(0, 1500, 15000)
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):

        name = self.spec_names[idx]

        smiles = self.name_to_smiles[name]
        mol_graph = self.name_to_mols[name]
        spec_form_obj = self.name_to_forms[name]

        full_formula = spec_form_obj["root_form"]
        form_list = spec_form_obj["formulae"]

        intens = spec_form_obj["intens"]
        adduct = self.adducts[idx]

        # Make sure it has at least 1..
        # (Patch up bug here)
        if len(form_list) == 0:
            form_list.append(full_formula)
            intens.append(0)
        ##

        form_intens = np.array(intens)

        # Get dense vec of all the formulae
        all_form_vecs = [common.formula_to_dense(i) for i in form_list]
        all_form_vecs = np.array(all_form_vecs)
        full_vec = common.formula_to_dense(full_formula)
        num_options = len(form_intens)
        diffs = full_vec - all_form_vecs

        if self.binned_targs:
            intens = np.array(spec_form_obj["raw_spec"])
            bin_posts = np.digitize(intens[:, 0], self.bins)
            new_out = np.zeros_like(self.bins)
            for bin_post, inten in zip(bin_posts, intens[:, 1]):
                new_out[bin_post] = max(new_out[bin_post], inten)
            out_inten = new_out
        else:
            out_inten = form_intens

        # print("DEBUGGING CLIPPING FORMS")
        # max_forms = np.random.randint(0, 200)
        # all_form_vecs = all_form_vecs[:max_forms]
        # form_intens = form_intens[:max_forms]
        # diffs = diffs[:max_forms]
        # num_options = max_forms
        ######

        outdict = {
            # Inputs
            "name": name,
            "smiles": smiles,
            "form_strs": form_list,
            "mol_graph": mol_graph,
            "base_formula": full_vec,
            "formulae": all_form_vecs,
            "diffs": diffs,
            "num_options": num_options,
            "targ_intens": out_inten,
            "adduct": adduct,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return IntenDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        mol_graphs = [j["mol_graph"] for j in input_list]
        form_strs = [j["form_strs"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        if isinstance(mol_graphs[0], dgl.DGLGraph):
            batched_graph = dgl.batch(mol_graphs)
        elif isinstance(mol_graphs[0], np.ndarray):
            batched_graph = torch.FloatTensor(np.vstack(mol_graphs))
        elif isinstance(mol_graphs[0], pyg_data):
            batched_graph = mformer.MassformerGraphFeaturizer.collate_func(mol_graphs)
        else:
            raise NotImplementedError()

        # Compute padding
        len_forms = np.array([j["num_options"] for j in input_list])
        pad_amts = max(len_forms) - len_forms

        # Pad formuale and intens
        formulae = [j["formulae"] for j in input_list]
        diffs = [j["diffs"] for j in input_list]
        formulae = [
            torch.nn.functional.pad(torch.FloatTensor(i), (0, 0, 0, pad_amt))
            for pad_amt, i in zip(pad_amts, formulae)
        ]
        diffs = [
            torch.nn.functional.pad(torch.FloatTensor(i), (0, 0, 0, pad_amt))
            for pad_amt, i in zip(pad_amts, diffs)
        ]

        # Deal with intens differentely in case there's a pad inten
        inten_ars = [j["targ_intens"] for j in input_list]
        inten_lens = [j.shape[0] for j in inten_ars]
        inten_pads = max(inten_lens) - np.array(inten_lens)
        intens = [
            torch.nn.functional.pad(torch.FloatTensor(i), (0, pad_amt))
            for pad_amt, i in zip(inten_pads, inten_ars)
        ]

        stacked_forms = torch.stack(formulae, 0)
        stacked_diffs = torch.stack(diffs, 0)
        stacked_intens = torch.stack(intens, 0)
        len_forms = torch.LongTensor(len_forms)

        adducts = torch.FloatTensor(adducts)
        output = {
            "names": names,
            "graphs": batched_graph,
            "form_strs": form_strs,
            "formulae": stacked_forms,
            "diffs": stacked_diffs,
            "intens": stacked_intens,
            "num_forms": len_forms,
            "adducts": adducts,
        }
        return output


class ScarfDataset(Dataset):
    """ScarfDataset."""

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
        """__init__ _summary_

        Args:
            df (pd.DataFrame): _description_
            data_dir (Path): _description_
            graph_featurizer (_type_): _description_
            file_map (dict): _description_
            num_workers (int, optional): _description_. Defaults to 0.
            root_embedder (str, optional): _description_. Defaults to "gnn".
            use_ray (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
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
            self.name_to_dict = self.df_sub.set_index("spec").to_dict(orient="index")

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
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.root_encode_fn,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        logging.info(f"{len(self.df_sub)} of {len(self.df)} spec have form dicts.")
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
        atom_inds, options, diffs, targ_intens = [], [], [], []

        all_prev_opts = []
        suffix_intens = []

        # Loop over each nonzero atom in the full formula and build progressive
        # targs
        # Input options are all_opts and output preds should be new_intens
        # Next can we compute the next set of options and output targets?
        # Need to define: 1. New options 2. Prev options 3. Add outter of the
        # two and 4. the intensities at each of these
        for ind, next_ind in enumerate(nonzero_inds):

            # New options
            # (A,)
            onehot = common.ELEMENT_VECTORS[next_ind]
            # (C,)
            new_options = np.arange(0, full_vec[next_ind] + 1)
            # (C,A)
            new_opts = onehot[None, :] * new_options[:, None]

            # (C_i,  C_{i-1}, A)
            all_opts = prev_opts[None, :, :] + new_opts[:, None, :]

            # (C_i * C_{i-1}, A)
            all_opts = all_opts.reshape(-1, atom_dim)
            new_diffs = full_vec - prev_opts

            # Calculate the intensities at each of these
            new_intens = np.zeros(all_opts.shape[0])
            for i, inten in zip(all_form_vecs, form_intens):
                # find where the formula i matches the options up to next_ind
                proper_ind = all_opts[:, : next_ind + 1] == i[: next_ind + 1]
                proper_ind = np.all(proper_ind, -1)
                new_intens[proper_ind] += inten

            new_intens_reshaped = np.zeros((prev_opts.shape[0], self.max_atom_ct))
            for frag_form, inten in zip(all_form_vecs, form_intens):

                # Define bool array
                proper_ind_prev = prev_opts[:, :next_ind] == frag_form[:next_ind]
                proper_ind_prev = np.all(proper_ind_prev, -1)

                next_ind_pos = int(frag_form[next_ind])

                # Define ind array for indexing
                proper_ind_prev_num = np.argwhere(proper_ind_prev).flatten()
                new_intens_reshaped[proper_ind_prev_num, next_ind_pos] += inten

            suffix_intens.append(new_intens_reshaped)

            # (C_{i - 1}, A)
            all_prev_opts.append(prev_opts)

            # Define (C_i, A) for next run through
            prev_opts = all_opts[new_intens > 0]

            # Store it all
            atom_inds.append(next_ind)
            targ_intens.append(new_intens)

            options.append(all_opts)
            diffs.append(new_diffs)

        # Create full dictionary outupt
        # option_lens = np.array([i.shape[0] for i in options])
        option_lens = np.array([i.shape[0] for i in all_prev_opts])
        atom_inds = np.array(atom_inds)

        # Outputs should have:
        # Name & smiles
        # Repr of molecule (graph)
        # Base formula
        # All options for decoding
        # How many options there are
        # Intensities of those
        # Which atom indices we are decoding at each option
        outdict = {
            # Inputs
            "name": name,
            "smiles": smiles,
            "mol_graph": mol_graph,
            "base_formula": full_vec,
            # Targets
            "option_len": option_lens,
            "options": all_prev_opts,  # options,
            "diffs": diffs,
            "atom_inds": atom_inds,
            "targ_intens": suffix_intens,  # targ_intens
            "adduct": adduct,
            # Added
            # "targ_len": option_lens,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return ScarfDataset.collate_fn

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
            batched_graph = mformer.MassformerGraphFeaturizer.collate_func(mol_graphs)
        else:
            raise NotImplementedError()

        formula_ars = [j["base_formula"] for j in input_list]

        # Expand out all examples
        atom_inds_expanded = [j for i in input_list for j in i["atom_inds"]]
        options_expanded = [j for i in input_list for j in i["options"]]
        diffs_expanded = [j for i in input_list for j in i["diffs"]]
        option_len_expanded = [j for i in input_list for j in i["option_len"]]
        targ_inten_expanded = [j for i in input_list for j in i["targ_intens"]]

        atom_inds_expanded = torch.LongTensor(atom_inds_expanded)
        formula_tensors = torch.stack(
            [torch.tensor(formula_ar) for formula_ar in formula_ars]
        )

        # How many atom types are in each molecule
        num_atom_types = torch.LongTensor([len(i["atom_inds"]) for i in input_list])

        # Maps molecule / formula to it respective options
        mol_inds = torch.arange(len(mol_graphs))
        mol_inds = mol_inds.repeat_interleave(num_atom_types, 0)

        # Computing padding
        max_opt_len = np.max(option_len_expanded)
        padding_amt = max_opt_len - np.array(option_len_expanded)
        option_len_expanded = torch.LongTensor(option_len_expanded)

        # Pad options, option_len, targ_inten, atom_ind
        options_expanded = [
            torch.nn.functional.pad(torch.LongTensor(i), (0, 0, 0, pad_amt))
            for pad_amt, i in zip(padding_amt, options_expanded)
        ]
        options_expanded = torch.stack(options_expanded)

        diffs_expanded = [
            torch.nn.functional.pad(torch.LongTensor(i), (0, 0, 0, pad_amt))
            for pad_amt, i in zip(padding_amt, diffs_expanded)
        ]
        diffs_expanded = torch.stack(diffs_expanded)
        targ_inten_expanded = [
            torch.nn.functional.pad(torch.FloatTensor(i), (0, 0, 0, pad_amt))
            for pad_amt, i in zip(padding_amt, targ_inten_expanded)
        ]
        targ_inten_expanded = torch.stack(targ_inten_expanded, 0)

        # Outputs should have:
        #  Name
        #  Fingerprint of actual molecule
        #  Full formula tensors
        #  Options
        #  Diffs
        #  Length of options
        #  intensities
        #  Atom indices we are working with
        #  How many atom types are for each molecule
        #  Mapping indices between the molecule and formulas and the options,
        # since each molecule correspodns to multiple option sets
        output = {
            "names": names,
            "graphs": batched_graph,
            "formula_tensors": formula_tensors,
            # Prefixes: (B * Depth) * (Max option len) * Atom
            "options": options_expanded,
            "diffs": diffs_expanded,
            # Num suffixes: (B * depth)
            "option_len": option_len_expanded,
            # Targ intensiites: (B * depth x max option len x max atom count)
            "targ_inten": targ_inten_expanded,
            # Which tree depth to compute: (B * depth)
            "atom_inds": atom_inds_expanded,
            # B
            "num_atom_types": num_atom_types,
            "mol_inds": mol_inds,
            "adducts": adducts,
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
