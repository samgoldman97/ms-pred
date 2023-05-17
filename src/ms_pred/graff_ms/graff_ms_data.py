from collections import defaultdict
import logging
import json
import numpy as np
from tqdm import tqdm

import torch
from rdkit import Chem
from torch.utils.data.dataset import Dataset
import dgl

import ms_pred.common as common

def array_to_string(array):
    """
    Converts a 1D NumPy array into a string.
    
    Args:
    	array: 1D NumPy array
    
    Returns:
		str_array: String representation of the array
    """
    str_array = np.array2string(array, separator=',')
    return str_array


def string_to_array(str_array):
    """
    Converts a string representation of a 1D NumPy array back into the array.
    
    Args:
		str_array: String representation of the array
    
    Returns:
    	array: 1D NumPy array
    """
    array = np.fromstring(str_array[1:-1], sep=',')
    return array


def extract_single_dict(entry):
    root = entry['root_form']
    intens, new_forms = entry['intens'], entry['formulae']
    freqs = defaultdict(lambda: 0)
    dense_forms = np.vstack([common.formula_to_dense(i) for i in new_forms])
    base_form = common.formula_to_dense(root)
    diffs = -(base_form[None, :] - dense_forms)

    for i, inten in zip(diffs, intens):
        freqs[array_to_string(i)] += inten

    for i, inten in zip(dense_forms, intens):
        freqs[array_to_string(i)] += inten

    return freqs

def merge_diffs(diff_list):
    """merge_diffs.

    Args:
        diff_list:
    """

    freq_diffs = defaultdict(lambda: 0)
    for freq_diff in tqdm(diff_list):
        for k, v in freq_diff.items():
            freq_diffs[k] += v
    return freq_diffs


def process_form_file(
    form_dict_file,
    upper_limit,
    num_bins
):
    """process_form_file."""
    if form_dict_file is None or not form_dict_file.exists():
        return None
    form_dict = json.load(open(form_dict_file, "r"))
    root_form = form_dict["cand_form"]
    out_tbl = form_dict["output_tbl"]

    if out_tbl is None:
        return None

    intens = out_tbl.get("ms2_inten", [])
    formulae = out_tbl.get("formula", [])

    # Get raw spec
    raw_spec = np.array(out_tbl.get("raw_spec"))

    bins = np.linspace(0, upper_limit, num_bins)
    bin_posts = np.digitize(raw_spec[:, 0], bins)
    new_out = np.zeros_like(bins)
    for bin_post, inten in zip(bin_posts, raw_spec[:, 1]):
        new_out[bin_post] = max(new_out[bin_post], inten)
    out_inten = new_out

    out_dict = dict(
        root_form=root_form, intens=intens, formulae=formulae, 
        raw_spec=raw_spec, raw_binned=out_inten
    )
    return out_dict


class BinnedDataset(Dataset):
    """SmiDataset."""

    def __init__(
        self,
        df,
        form_map,
        data_dir,
        num_bins,
        graph_featurizer,
        num_workers=0,
        upper_limit=1500,
        form_dir_name: str = "subform_20",
        use_ray=False,
        **kwargs,
    ):
        self.df = df
        self.num_bins = num_bins
        self.file_map = form_map
        self.num_workers = num_workers
        self.upper_limit = upper_limit
        self.bins = np.linspace(0, self.upper_limit, self.num_bins)
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

        # Read in all molecules
        self.smiles = self.df_sub["smiles"].values
        self.graph_featurizer = graph_featurizer
        self.num_atom_feats = self.graph_featurizer.num_atom_feats
        self.num_bond_feats = self.graph_featurizer.num_bond_feats

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.weights = [common.ExactMolWt(i) for i in self.mols]
            self.mol_graphs = [
                self.graph_featurizer.get_dgl_graph(i) for i in self.mols
            ]
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
                self.graph_featurizer.get_dgl_graph,
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
        self.weights = np.array(self.weights)

        # Read in all specs
        spec_files = self.form_files
        self.spec_names = self.df_sub["spec"].values
        wrapper_process = lambda x: process_form_file(
            x, num_bins=num_bins, upper_limit=upper_limit
        )

        if self.num_workers == 0:
            spec_outputs = [wrapper_process(i) for i in spec_files]
        else:
            spec_outputs = common.chunked_parallel(
                spec_files,
                wrapper_process,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        self.name_to_forms = dict(zip(self.spec_names, spec_outputs))
        self.name_to_smiles = dict(zip(self.spec_names, self.smiles))
        self.name_to_mols = dict(zip(self.spec_names, self.mol_graphs))

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


    def get_top_forms(self): 
        all_freqs = []
        for i in self.spec_names:
            entry = self.name_to_forms[i]
            extracted = extract_single_dict(entry)
            all_freqs.append(extracted)
        freq_diff = merge_diffs(all_freqs)
        forms, cts = zip(*list(freq_diff.items()))
        cts = np.array(cts)
        forms = np.array([string_to_array(i) for i in forms])
        new_order = np.argsort(cts)[::-1]
        cts = cts[new_order]
        forms = forms[new_order]
        return {"forms": forms, "cts": cts}


    def __len__(self):
        return len(self.spec_names)


    def __getitem__(self, idx: int):
        name = self.spec_names[idx]

        smiles = self.name_to_smiles[name]
        graph = self.name_to_mols[name]
        spec_form_obj = self.name_to_forms[name]

        ar = spec_form_obj['raw_binned']
        adduct = self.adducts[idx]

        outdict = {
            "name": name,
            "root_form": common.formula_to_dense(spec_form_obj['root_form']),
            "binned": ar,
            "adduct": adduct,
            "graph": graph,
            "smiles": smiles
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
        graphs = [j["graph"] for j in input_list]
        full_forms = [j["root_form"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        # Now pad everything else to the max channel dim
        spectra_tensors = torch.stack([torch.tensor(spectra) for spectra in spec_ars])

        batched_graph = dgl.batch(graphs)
        # frag_batch.set_n_initializer(dgl.init.zero_initializer)
        # frag_batch.set_e_initializer(dgl.init.zero_initializer)

        adducts = torch.FloatTensor(adducts)
        full_forms = torch.FloatTensor(full_forms)

        return_dict = {
            "spectra": spectra_tensors,
            "graphs": batched_graph,
            "full_forms": full_forms,
            "names": names,
            "adducts": adducts,
        }
        return return_dict


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(self, df, graph_featurizer, num_workers: int = 0, **kwargs):

        self.df = df
        self.num_workers = num_workers
        self.graph_featurizer = graph_featurizer
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        self.spec_names = ["" for i in self.smiles]
        if "spec" in list(self.df.keys()):
            self.spec_names = self.df["spec"].values

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.weights = [common.ExactMolWt(i) for i in self.mols]
            self.mol_graphs = [
                self.graph_featurizer.get_dgl_graph(i) for i in self.mols
            ]
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
                self.graph_featurizer.get_dgl_graph,
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
        root_form = common.formula_to_dense(common.form_from_smi(smi))
        graph = self.mol_graphs[idx]
        spec_name = self.spec_names[idx]
        adduct = self.adducts[idx]
        outdict = {
            "smi": smi,
            "graph": graph,
            "root_form": root_form,
            "adduct": adduct,
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
        graphs = [j["graph"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)

        full_forms = torch.FloatTensor([j['root_form'] for j in input_list])
        batched_graph = dgl.batch(graphs)
        return_dict = {
            "graphs": batched_graph,
            "spec_names": spec_names,
            "names": names,
            "full_forms": full_forms,
            "adducts": adducts,
        }
        return return_dict
