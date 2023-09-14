""" dag_data.py

Fragment dataset to build out model class

"""
import logging
from pathlib import Path
from typing import List
import json
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import dgl
from torch.utils.data.dataset import Dataset

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation


class TreeProcessor:
    """TreeProcessor.

    Hold key functionalities to read in a magma dag and proces it.

    """

    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
    ):
        """ """
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs

        # Hard coded bins (for now)
        self.bins = np.linspace(0, 1500, 15000)

    def featurize_frag(
        self,
        frag: int,
        engine: fragmentation.FragmentEngine,
        add_random_walk: bool = False,
    ) -> False:
        """featurize_frag.

        Prev.  dgl_from_frag

        """

        num_atoms = engine.natoms
        atom_symbols = engine.atom_symbols
        # Need to find all kept atoms and all kept bonds between
        kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
        kept_bond_orders, kept_bonds = engine.get_present_edges(frag)

        # H count
        form = engine.formula_from_kept_inds(kept_atom_inds)

        # Need to re index the targets to match the new graph size
        num_kept = len(kept_atom_inds)
        new_inds = np.arange(num_kept)
        old_inds = kept_atom_inds

        old_to_new = np.zeros(num_atoms, dtype=int)
        old_to_new[old_inds] = new_inds

        # Keep new_to_old for autoregressive predictions
        new_to_old = np.zeros(num_kept, dtype=int)
        new_to_old[new_inds] = old_inds

        # Remap new bond inds
        new_bond_inds = np.empty((0, 2), dtype=int)
        if len(kept_bonds) > 0:
            new_bond_inds = old_to_new[np.vstack(kept_bonds)]

        if self.add_hs:
            h_adds = np.array(engine.atom_hs)[kept_atom_inds]
        else:
            h_adds = None

        # Make dgl graphs for new targets
        graph = self.dgl_featurize(
            np.array(atom_symbols)[kept_atom_inds],
            h_adds=h_adds,
            bond_inds=new_bond_inds,
            bond_types=np.array(kept_bond_orders),
        )

        if add_random_walk:
            self.add_pe_embed(graph)
        return {
            "graph": graph,
            "new_to_old": new_to_old,
            "old_to_new": old_to_new,
            "form": form,
        }

    def _convert_to_dgl(
        self,
        tree: dict,
        include_targets: bool = True,
        last_row=False,
    ):
        """_convert_to_dgl.

        Args:
            tree (dict): tree dictionary
            include_targets (bool): Try to add inten targets for supervising
                the inten model
            last_row:
        """
        root_inchi = tree["root_inchi"]
        engine = fragmentation.FragmentEngine(mol_str=root_inchi, mol_str_type="inchi")
        # bottom_depth = engine.max_broken_bonds
        bottom_depth = engine.max_tree_depth
        if self.root_encode == "gnn":
            root_frag = engine.get_root_frag()
            root_graph_dict = self.featurize_frag(
                frag=root_frag,
                engine=engine,
            )
            root_repr = root_graph_dict["graph"]
        elif self.root_encode == "fp":
            root_repr = common.get_morgan_fp_inchi(root_inchi)
        else:
            raise ValueError()

        root_form = common.form_from_inchi(root_inchi)

        # Need to include mass and inten targets here, maybe not necessary in
        # all cases?
        masses, inten_frag_ids, dgl_inputs, inten_targets, frag_targets, max_broken = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        forms = []
        max_remove_hs, max_add_hs = [], []
        for k, sub_frag in tree["frags"].items():
            max_broken_num = sub_frag["max_broken"]
            tree_depth = sub_frag["tree_depth"]

            # Skip because we never fragment last row
            if (not last_row) and (tree_depth == bottom_depth):
                continue

            binary_targs = sub_frag["atoms_pulled"]
            frag = sub_frag["frag"]

            # Get frag dict and target
            frag_dict = self.featurize_frag(
                frag,
                engine,
            )
            forms.append(frag_dict["form"])
            old_to_new = frag_dict["old_to_new"]
            graph = frag_dict["graph"]
            max_broken.append(max_broken_num)

            max_remove_hs.append(sub_frag["max_remove_hs"])
            max_add_hs.append(sub_frag["max_add_hs"])

            if include_targets and not self.binned_targs:
                inten_targs = sub_frag["intens"]
                inten_targets.append(inten_targs)

            inten_frag_ids.append(k)

            # For gen model only!!
            targ_vec = np.zeros(graph.num_nodes())
            for j in old_to_new[binary_targs]:
                targ_vec[j] = 1

            graph = frag_dict["graph"]

            # Define targ vec
            dgl_inputs.append(graph)
            masses.append(sub_frag["base_mass"])
            frag_targets.append(torch.from_numpy(targ_vec))

        if include_targets and self.binned_targs:
            inten_targets = np.array(tree["raw_spec"])

        masses = engine.shift_bucket_masses[None, :] + np.array(masses)[:, None]
        max_remove_hs = np.array(max_remove_hs)
        max_add_hs = np.array(max_add_hs)
        max_broken = np.array(max_broken)

        # Feat each form
        all_form_vecs = [common.formula_to_dense(i) for i in forms]
        all_form_vecs = np.array(all_form_vecs)
        root_form_vec = common.formula_to_dense(root_form)

        out_dict = {
            "root_repr": root_repr,
            "dgl_frags": dgl_inputs,
            "masses": masses,
            "inten_targs": np.array(inten_targets) if include_targets else None,
            "inten_frag_ids": inten_frag_ids,
            "max_remove_hs": max_remove_hs,
            "max_add_hs": max_add_hs,
            "max_broken": max_broken,
            "targs": frag_targets,
            "form_vecs": all_form_vecs,
            "root_form_vec": root_form_vec,
        }
        return out_dict

    def _process_tree(
        self,
        tree: dict,
        include_targets: bool = True,
        last_row=False,
        convert_to_dgl=True,
    ):
        """_process_tree.

        Args:
            tree (dict): tree dictionary
            include_targets (bool): Try to add inten targets for supervising
                the inten model
            last_row:
            pickle_input: If pickle_input, this
        """
        if convert_to_dgl:
            out_dict = self._convert_to_dgl(tree, include_targets, last_row)
        else:
            out_dict = tree

        dgl_inputs = out_dict["dgl_frags"]
        root_repr = out_dict["root_repr"]

        if self.pe_embed_k > 0:
            for graph in dgl_inputs:
                self.add_pe_embed(graph)

            if isinstance(root_repr, dgl.DGLGraph):
                self.add_pe_embed(root_repr)

        if include_targets and self.binned_targs:
            intens = out_dict["inten_targs"]
            bin_posts = np.digitize(intens[:, 0], self.bins)
            new_out = np.zeros_like(self.bins)
            for bin_post, inten in zip(bin_posts, intens[:, 1]):
                new_out[bin_post] = max(new_out[bin_post], inten)
            inten_targets = new_out
            out_dict["inten_targs"] = inten_targets
        return out_dict

    def add_pe_embed(self, graph):
        pe_embeds = nn_utils.random_walk_pe(
            graph, k=self.pe_embed_k, eweight_name="e_ind"
        )
        graph.ndata["h"] = torch.cat((graph.ndata["h"], pe_embeds), -1).float()
        return graph

    def process_tree_gen(self, tree: dict, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=False, last_row=False, convert_to_dgl=convert_to_dgl
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "targs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten(self, tree, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=True, last_row=True, convert_to_dgl=convert_to_dgl
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_targs",
            "inten_frag_ids",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten_pred(self, tree: dict, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=False, last_row=True, convert_to_dgl=convert_to_dgl
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_targs",
            "inten_frag_ids",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def dgl_featurize(
        self,
        atom_symbols: List[str],
        h_adds: np.ndarray,
        bond_inds: np.ndarray,
        bond_types: np.ndarray,
    ):
        """dgl_featurize.

        Args:
            atom_symbols (List[str]): node_types
            h_adds (np.ndarray): h_adds
            bond_inds (np.ndarray): bond_inds
            bond_types (np.ndarray)
        """
        node_types = [common.element_to_position[el] for el in atom_symbols]
        node_types = np.vstack(node_types)
        num_nodes = node_types.shape[0]

        src, dest = bond_inds[:, 0], bond_inds[:, 1]
        src_tens_, dest_tens_ = torch.from_numpy(src), torch.from_numpy(dest)
        bond_types = torch.from_numpy(bond_types)
        src_tens = torch.cat([src_tens_, dest_tens_])
        dest_tens = torch.cat([dest_tens_, src_tens_])
        bond_types = torch.cat([bond_types, bond_types])
        bond_featurizer = torch.eye(fragmentation.MAX_BONDS)

        bond_types_onehot = bond_featurizer[bond_types.long()]
        node_data = torch.from_numpy(node_types)

        # H data is defined, add that
        if h_adds is None:
            zero_vec = torch.zeros((node_data.shape[0], common.MAX_H))
            node_data = torch.hstack([node_data, zero_vec])
        else:
            h_featurizer = torch.eye(common.MAX_H)
            h_adds_vec = torch.from_numpy(h_adds)
            node_data = torch.hstack([node_data, h_featurizer[h_adds_vec]])

        g = dgl.graph(data=(src_tens, dest_tens), num_nodes=num_nodes)
        g.ndata["h"] = node_data.float()
        g.edata["e"] = bond_types_onehot.float()
        g.edata["e_ind"] = bond_types.long()
        return g

    def get_node_feats(self):
        return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H


class DAGDataset(Dataset):
    """DAGDataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            df (pd.DataFrame): df
            data_dir (Path): data_dir
            magma_map (dict): magma_map
            num_workers:
            use_ray (bool): use_ray
            kwargs:
        """
        self.df = df
        self.num_workers = num_workers
        self.magma_map = magma_map

        valid_specs = [i in self.magma_map for i in self.df["spec"].values]
        self.df_sub = self.df[valid_specs]
        if len(self.df_sub) == 0:
            self.spec_names = []
            self.name_to_dict = {}
        else:
            self.spec_names = self.df_sub["spec"].values
            self.name_to_dict = self.df_sub.set_index("spec").to_dict(orient="index")

        for i in self.name_to_dict:
            self.name_to_dict[i]["magma_file"] = self.magma_map[i]

        logging.info(f"{len(self.df_sub)} of {len(self.df)} spec have trees.")
        read_fn = lambda x: self.read_tree(self.load_tree(x))
        self.read_fn = read_fn

        # Read in all trees necessary in parallel
        # Process trees jointly; sacrifice mmeory for speed
        if False:
            if self.num_workers > 0:
                tree_outs = common.chunked_parallel(
                    self.spec_names,
                    read_fn,
                    max_cpu=self.num_workers,
                    chunks=min(len(self.spec_names) // 4, 50),
                    timeout=1200,
                    use_ray=use_ray,
                )
                self.trees = [i["tree"] for i in tree_outs]
                self.dgl_trees = [i["dgl_tree"] for i in tree_outs]

            else:
                tree_outs = [read_fn(i) for i in tqdm(self.spec_names)]
                self.trees = [i["tree"] for i in tree_outs]
                self.dgl_trees = [i["dgl_tree"] for i in tree_outs]
            self.spec_name_to_tree = dict(zip(self.spec_names, self.trees))

        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)
        self.name_to_adducts = {
            i: common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        }
        self.name_to_smiles = dict(self.df[["spec", "smiles"]].values)

    def load_tree(self, x):
        filename = self.name_to_dict[x]["magma_file"]
        with open(str(filename), "r") as fp:
            return json.load(fp)

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):
        """__getitem__."""

        # name = copy.deepcopy(self.spec_names[idx])
        # adduct = copy.deepcopy(self.name_to_adducts[name])
        # dgl_entry = copy.deepcopy(self.dgl_trees[idx])

        name = self.spec_names[idx]
        adduct = self.name_to_adducts[name]

        dgl_entry = self.read_fn(name)["dgl_tree"]
        # dgl_entry = self.dgl_trees[idx]

        outdict = {"name": name, "adduct": adduct}

        # Convert this into a list of graphs with a list of targets
        outdict.update(dgl_entry)
        return outdict

    def get_node_feats(self) -> int:
        """get_node_feats."""
        return self.tree_processor.get_node_feats()


class GenDataset(DAGDataset):
    """GenDatset."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        tree_processor: TreeProcessor,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
        """
        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.process_tree_gen
        super().__init__(
            df=df,
            data_dir=data_dir,
            magma_map=magma_map,
            num_workers=num_workers,
            use_ray=use_ray,
        )

    @classmethod
    def get_collate_fn(cls):
        return GenDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn.

        Batch a list of graphs and convert.

        Should return dict containing:
            1. Vector of names.
            2. Batched root reprs
            3. Batched fragment graphs
            4. Batched target atoms
            6. Number of atoms per frag graph
            7. Indices mapping batched fragment graphs to roots
            8. Conversion mapping indices between root and children?

        """
        names = [j["name"] for j in input_list]
        frag_graphs = [j["dgl_frags"] for j in input_list]
        frag_graphs_e = [j for i in frag_graphs for j in i]

        num_frags = torch.LongTensor([len(i) for i in frag_graphs])
        frag_atoms = torch.LongTensor([j.num_nodes() for i in frag_graphs for j in i])

        targs = [j for i in input_list for j in i["targs"]]
        targs_padded = torch.nn.utils.rnn.pad_sequence(targs, batch_first=True)

        batched_reprs = _collate_root(input_list)

        frag_batch = dgl.batch(frag_graphs_e)
        root_inds = torch.arange(len(frag_graphs)).repeat_interleave(num_frags)

        max_broken = [torch.LongTensor(i["max_broken"]) for i in input_list]
        max_broken = torch.cat(max_broken)
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)

        # Collate forms
        form_vecs = torch.LongTensor([j for i in input_list for j in i["form_vecs"]])
        root_vecs = torch.LongTensor([i["root_form_vec"] for i in input_list])
        output = {
            "names": names,
            "root_reprs": batched_reprs,
            "frag_graphs": frag_batch,
            "targ_atoms": targs_padded,
            "frag_atoms": frag_atoms,
            "inds": root_inds,
            "broken_bonds": max_broken,
            "adducts": adducts,
            "root_form_vecs": root_vecs,
            "frag_form_vecs": form_vecs,
        }
        return output


class IntenDataset(DAGDataset):
    """GenDatset."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        tree_processor: TreeProcessor,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
        """
        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.process_tree_inten
        super().__init__(
            df=df,
            data_dir=data_dir,
            magma_map=magma_map,
            num_workers=num_workers,
            use_ray=use_ray,
        )

    @classmethod
    def get_collate_fn(cls):
        return IntenDataset.collate_fn

    @staticmethod
    def collate_fn(
        input_list,
    ):
        """collate_fn.

        Batch a list of graphs and convert.

        Should return dict containing:
            1. Vector of names.
            2. Batched root graphs
            3. Batched fragment graphs
            5. Number of atoms per frag graph
            6. Conversion mapping indices between root and children
            7. Inten targets
            8. Num broken bonds per atom
            9. Num frags in each mols dag

        """
        names = [j["name"] for j in input_list]
        frag_graphs = [j["dgl_frags"] for j in input_list]
        frag_graphs_e = [j for i in frag_graphs for j in i]
        num_frags = torch.LongTensor([len(i) for i in frag_graphs])
        frag_atoms = torch.LongTensor([i.num_nodes() for i in frag_graphs_e])
        batched_reprs = _collate_root(input_list)

        frag_batch = dgl.batch(frag_graphs_e)
        root_inds = torch.arange(len(frag_graphs)).repeat_interleave(num_frags)

        targs_padded = None
        if "inten_targs" in input_list[0] and input_list[0]["inten_targs"] is not None:

            if len(input_list[0]["inten_targs"].shape) == 1:
                targs_padded = torch.FloatTensor(
                    np.vstack([i["inten_targs"] for i in input_list])
                )
            else:
                targs_padded = _unroll_pad(input_list, "inten_targs")

        masses_padded = _unroll_pad(input_list, "masses")
        max_remove_hs_padded = _unroll_pad(input_list, "max_remove_hs")
        max_add_hs_padded = _unroll_pad(input_list, "max_add_hs")

        inten_frag_ids = None
        if input_list[0]["inten_frag_ids"] is not None:
            inten_frag_ids = [i["inten_frag_ids"] for i in input_list]

        max_broken = [torch.LongTensor(i["max_broken"]) for i in input_list]
        broken_padded = torch.nn.utils.rnn.pad_sequence(max_broken, batch_first=True)

        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)

        # Collate fomrs
        form_vecs = _unroll_pad(input_list, "form_vecs")
        root_vecs = _unroll_pad(input_list, "root_form_vec")

        output = {
            "names": names,
            "root_reprs": batched_reprs,
            "frag_graphs": frag_batch,
            "frag_atoms": frag_atoms,
            "inds": root_inds,
            "num_frags": num_frags,
            "inten_targs": targs_padded,
            "masses": masses_padded,
            "broken_bonds": broken_padded,
            "max_add_hs": max_add_hs_padded,
            "max_remove_hs": max_remove_hs_padded,
            "inten_frag_ids": inten_frag_ids,
            "adducts": adducts,
            "root_form_vecs": root_vecs,
            "frag_form_vecs": form_vecs,
        }
        return output


class IntenPredDataset(DAGDataset):
    """IntenPredDatset."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        tree_processor: TreeProcessor,
        magma_map: dict,
        num_workers=0,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            df (pd.DataFrame): _description_
            data_dir (Path): _description_
            tree_processor (TreeProcessor): _description_
            magma_map (dict): _description_
            num_workers (int, optional): _description_. Defaults to 0.
        """
        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.process_tree_inten_pred
        super().__init__(
            df=df,
            data_dir=data_dir,
            magma_map=magma_map,
            num_workers=num_workers,
            tree_processor=tree_processor,
        )

    @classmethod
    def get_collate_fn(cls):
        return IntenDataset.collate_fn


def _unroll_pad(input_list, key):
    if input_list[0][key] is not None:
        out = [torch.FloatTensor(i[key]) for i in input_list]
        out_padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
        return out_padded
    return None


def _collate_root(input_list):
    root_reprs = [j["root_repr"] for j in input_list]
    if isinstance(root_reprs[0], dgl.DGLGraph):
        batched_reprs = dgl.batch(root_reprs)
    elif isinstance(root_reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    else:
        raise NotImplementedError()
    return batched_reprs
