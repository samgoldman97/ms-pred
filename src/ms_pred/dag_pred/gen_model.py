"""frag_model."""
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import dgl
import dgl.nn as dgl_nn


import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as magma
import ms_pred.dag_pred.dag_data as dag_data


class FragGNN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        set_layers: int = 2,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        dropout: float = 0,
        mpnn_type: str = "GGNN",
        pool_op: str = "avg",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        pe_embed_k: int = 0,
        max_broken: int = magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        root_encode: str = "gnn",
        inject_early: bool = False,
        warmup: int = 1000,
        embed_adduct=False,
        encode_forms: bool = False,
        add_hs: bool = False,
        **kwargs,
    ):
        """__init__.
        Args:
            hidden_size (int): Hidden size
            layers (int): Num layers
            set_layers (int): Set layers
            learning_rate (float): Learning rate
            lr_decay_rate (float)
            weight_decay (float): amt of weight decay
            dropout (float): Dropout
            mpnn_type (str): Type of MPNN
            pool_op (str):
            node_feats
            pe_embed_k
            max_broken
            root_encode
            inject_early (bool)
            warmup
            add_hs
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.root_encode = root_encode
        self.pe_embed_k = pe_embed_k
        self.embed_adduct = embed_adduct
        self.encode_forms = encode_forms
        self.add_hs = add_hs

        self.tree_processor = dag_data.TreeProcessor(
            root_encode=root_encode, pe_embed_k=pe_embed_k, add_hs=self.add_hs
        )
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim

            # Account for diffs
            self.formula_in_dim *= 2

        self.pool_op = pool_op
        self.inject_early = inject_early

        self.layers = layers
        self.mpnn_type = mpnn_type
        self.set_layers = set_layers

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = torch.nn.Parameter(torch.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        edge_feats = fragmentation.MAX_BONDS

        orig_node_feats = node_feats
        if self.inject_early:
            node_feats = node_feats + self.hidden_size

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.layers,
            set_transform_layers=self.set_layers,
            mpnn_type=self.mpnn_type,
            gnn_node_feats=node_feats + adduct_shift,
            gnn_edge_feats=edge_feats,
            dropout=self.dropout,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn

            # if inject early, need separate root and child GNN's
            if self.inject_early:
                self.root_module = nn_utils.MoleculeGNN(
                    hidden_size=self.hidden_size,
                    num_step_message_passing=self.layers,
                    set_transform_layers=self.set_layers,
                    mpnn_type=self.mpnn_type,
                    gnn_node_feats=orig_node_feats + adduct_shift,
                    gnn_edge_feats=edge_feats,
                    dropout=self.dropout,
                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                use_residuals=True,
                num_layers=1,
            )
        else:
            raise ValueError()

        # MLP layer to take representations from the pooling layer
        # And predict a single scalar value at each of them
        # I.e., Go from size B x 2h -> B x 1
        self.output_map = nn_utils.MLPBlocks(
            input_size=self.hidden_size * 3 + self.max_broken + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            dropout=self.dropout,
            num_layers=1,
            use_residuals=True,
        )

        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        broken,
        adducts,
        root_forms=None,
        frag_forms=None,
    ):
        """predict spec from graphs

        graphs: DGL Graphs of all mols in batch
        root_repr: DGL graphs of roots for each or FP
        ind_maps: Mapping to pair graphs to root graphs (allows parallel)
        broken: number of broken bonds
        adducts
        root_forms
        frag_forms
        """
        # if root fingerprints:
        embed_adducts = self.adduct_embedder[adducts.long()]
        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            with root_repr.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = torch.cat([ndata, embed_adducts_expand], -1)
                    root_repr.ndata["h"] = ndata
                root_embeddings = self.root_module(root_repr)
                root_embeddings = self.pool(root_repr, root_embeddings)
        else:
            pass

        # Line up the features to be parallel between fragment avgs and root
        # graphs
        ext_root = root_embeddings[ind_maps]

        # Extend the root further to cover each individual atom
        ext_root_atoms = torch.repeat_interleave(
            ext_root, graphs.batch_num_nodes(), dim=0
        )
        concat_list = [graphs.ndata["h"]]

        if self.inject_early:
            concat_list.append(ext_root_atoms)

        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]
            adducts_exp = torch.repeat_interleave(
                adducts_mapped, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(adducts_exp)

        with graphs.local_scope():
            graphs.ndata["h"] = torch.cat(concat_list, -1).float()

            frag_embeddings = self.gnn(graphs)

            # Average embed the full root molecules and fragments
            avg_frags = self.pool(graphs, frag_embeddings)

        # Extend the avg of each fragment
        ext_frag_atoms = torch.repeat_interleave(
            avg_frags, graphs.batch_num_nodes(), dim=0
        )

        exp_num = graphs.batch_num_nodes()
        # Do the same with the avg fragments

        broken = torch.clamp(broken, max=self.broken_clamp)
        ext_frag_broken = torch.repeat_interleave(broken, exp_num, dim=0)
        broken_onehots = self.broken_onehot[ext_frag_broken.long()]

        mlp_cat_vec = [
            ext_root_atoms,
            ext_root_atoms - ext_frag_atoms,
            frag_embeddings,
            broken_onehots,
        ]
        if self.encode_forms:
            root_exp = root_forms[ind_maps]
            diffs = root_exp - frag_forms
            form_encodings = self.embedder(frag_forms)
            diff_encodings = self.embedder(diffs)
            form_atom_exp = torch.repeat_interleave(form_encodings, exp_num, dim=0)
            diff_atom_exp = torch.repeat_interleave(diff_encodings, exp_num, dim=0)

            mlp_cat_vec.extend([form_atom_exp, diff_atom_exp])

        hidden = torch.cat(
            mlp_cat_vec,
            dim=1,
        )

        output = self.output_map(hidden)
        output = self.sigmoid(output)
        padded_out = nn_utils.pad_packed_tensor(output, graphs.batch_num_nodes(), 0)
        padded_out = torch.squeeze(padded_out, -1)
        return padded_out

    def loss_fn(self, outputs, targets, natoms):
        """loss_fn.

        Args:
            outputs: Outputs after sigmoid fucntion
            targets: Target binary vals
            natoms: Number of atoms in each atom to consider padding

        """
        loss = self.bce_loss(outputs, targets.float())
        is_valid = (
            torch.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
        )
        pooled_loss = torch.sum(loss * is_valid) / torch.sum(natoms)
        return pooled_loss

    def _common_step(self, batch, name="train"):
        pred_leaving = self.forward(
            batch["frag_graphs"],
            batch["root_reprs"],
            batch["inds"],
            broken=batch["broken_bonds"],
            adducts=batch["adducts"],
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
        )
        loss = self.loss_fn(pred_leaving, batch["targ_atoms"], batch["frag_atoms"])
        self.log(
            f"{name}_loss", loss.item(), on_epoch=True, batch_size=len(batch["names"])
        )
        return loss

    def training_step(self, batch, batch_idx):
        """training_step."""
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        """validation_step."""
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        """test_step."""
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        """configure_optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }
        return ret

    def predict_mol(
        self,
        root_smi: str,
        adduct,
        threshold=0,
        device: str = "cpu",
        max_nodes: int = None,
    ) -> dict:
        """prdict_mol.

        Predict a new fragmentation tree from a starting root molecule
        autoregressively. First a new fragment is added to the
        frag_hash_to_entry dict and also put on the stack. Then it is
        fragmented and its "atoms_pulled" and "left_pred" are updated
        accordingly. The resulting new fragments are added to the hash.

        Args:
            root_smi (smi)
            threshold: Leaving probability
            device: Device
            max_nodes (int): Max number to include

        Return:
            Dictionary containing results
        """
        # Step 1: Get a fragmentation engine for root mol
        engine = fragmentation.FragmentEngine(root_smi)
        max_depth = engine.max_tree_depth
        root_frag = engine.get_root_frag()
        root_form = common.form_from_smi(root_smi)
        root_form_vec = torch.FloatTensor(common.formula_to_dense(root_form))
        root_form_vec = root_form_vec.reshape(1, -1)
        adducts = torch.LongTensor([common.ion2onehot_pos[adduct]])

        # Step 2: Featurize the root molecule
        root_graph_dict = self.tree_processor.featurize_frag(
            frag=root_frag,
            engine=engine,
            add_random_walk=True,
        )

        root_repr = None
        if self.root_encode == "gnn":
            root_repr = root_graph_dict["graph"].to(device)
        elif self.root_encode == "fp":
            root_fp = torch.from_numpy(np.array(common.get_morgan_fp_smi(root_smi)))
            root_repr = root_fp.float().to(device)[None, :]

        form_to_min_score = {}
        frag_hash_to_entry = {}
        frag_to_hash = {}
        stack = [root_frag]
        depth = 0
        root_hash = engine.wl_hash(root_frag)
        frag_to_hash[root_frag] = root_hash
        root_score = engine.score_fragment(root_frag)[1]
        id_ = 0
        # TODO: Compute as in fragment engine
        root_entry = {
            "frag": int(root_frag),
            "frag_hash": root_hash,
            "parents": [],
            "atoms_pulled": [],
            "left_pred": [],
            "max_broken": 0,
            "tree_depth": 0,
            "id": 0,
            "prob_gen": 1,
            "score": root_score,
        }
        id_ += 1
        root_entry.update(engine.atom_pass_stats(root_frag, depth=0))
        form_to_min_score[root_entry["form"]] = root_entry["score"]
        frag_hash_to_entry[root_hash] = root_entry

        # Step 3: Run the autoregressive gen loop
        with torch.no_grad():

            # Note: we don't fragment at the final depth
            while len(stack) > 0 and depth < max_depth:
                # Convert all new frags to graphs (stack is to run next)

                new_dgl_dicts = [
                    self.tree_processor.featurize_frag(
                        frag=i, engine=engine, add_random_walk=True
                    )
                    for i in stack
                ]
                tuple_list = [
                    (i, j)
                    for i, j in zip(new_dgl_dicts, stack)
                    if i["graph"].num_nodes() > 1
                ]

                if len(tuple_list) == 0:
                    break

                new_dgl_dicts, stack = zip(*tuple_list)
                mol_batch_graph = [i["graph"] for i in new_dgl_dicts]
                frag_forms = [i["form"] for i in new_dgl_dicts]
                frag_form_vecs = [common.formula_to_dense(i) for i in frag_forms]
                frag_form_vecs = torch.FloatTensor(np.array(frag_form_vecs))

                new_frag_hashes = [engine.wl_hash(i) for i in stack]

                frag_to_hash.update(dict(zip(stack, new_frag_hashes)))

                frag_batch = dgl.batch(mol_batch_graph).to(device)
                inds = torch.zeros(frag_batch.batch_size).long().to(device)

                # Note: Can speed by reducing redundant root graph passes
                # TODO: Figure out how to include frag form vec and root form
                # vec

                # TODO: Compute broken for each of these...

                broken_nums_ar = np.array(
                    [frag_hash_to_entry[i]["max_broken"] for i in new_frag_hashes]
                )
                broken_nums_tensor = torch.FloatTensor(broken_nums_ar).to(device)

                pred_leaving = self.forward(
                    graphs=frag_batch,
                    root_repr=root_repr,
                    ind_maps=inds,
                    broken=broken_nums_tensor,  # torch.ones_like(inds) * depth,
                    adducts=adducts,
                    root_forms=root_form_vec,
                    frag_forms=frag_form_vecs,
                )
                depth += 1

                # Rank order all the atom preds and predictions
                # Continuously add items to the stack as long as they maintain
                # the max node constraint ranked by prob

                # Get all frag probabilities and sort them
                cur_probs = sorted(
                    [i["prob_gen"] for i in frag_hash_to_entry.values()]
                )[::-1]
                if max_nodes is None or len(cur_probs) < max_nodes:
                    min_prob = threshold
                elif max_nodes is not None and len(cur_probs) >= max_nodes:
                    min_prob = cur_probs[max_nodes - 1]
                else:
                    raise NotImplementedError()

                new_items = list(
                    zip(stack, new_frag_hashes, pred_leaving, new_dgl_dicts)
                )
                sorted_order = []
                for item_ind, item in enumerate(new_items):
                    frag_hash = item[1]
                    pred_vals_f = item[2]
                    parent_prob = frag_hash_to_entry[frag_hash]["prob_gen"]
                    for atom_ind, (atom_pred, prob_gen) in enumerate(
                        zip(pred_vals_f, parent_prob * pred_vals_f)
                    ):

                        sorted_order.append(
                            dict(
                                item_ind=item_ind,
                                atom_ind=atom_ind,
                                prob_gen=prob_gen.item(),
                                atom_pred=atom_pred.item(),
                            )
                        )

                sorted_order = sorted(sorted_order, key=lambda x: -x["prob_gen"])
                new_stack = []

                # Process ordered list continuously
                for new_item in sorted_order:
                    prob_gen = new_item["prob_gen"]
                    atom_ind = new_item["atom_ind"]
                    atom_pred = new_item["atom_pred"]
                    item_ind = new_item["item_ind"]

                    # Filter out on minimum prob
                    if prob_gen <= min_prob:
                        continue

                    # Calc stack ind
                    orig_entry = new_items[item_ind]
                    frag_int = orig_entry[0]
                    frag_hash = orig_entry[1]
                    dgl_dict = orig_entry[3]

                    # Get atom ind
                    atom = dgl_dict["new_to_old"][atom_ind]

                    # Calc remove dict
                    out_dicts = engine.remove_atom(frag_int, int(atom))

                    # Update atoms_pulled for parent
                    frag_hash_to_entry[frag_hash]["atoms_pulled"].append(int(atom))
                    frag_hash_to_entry[frag_hash]["left_pred"].append(float(atom_pred))
                    parent_broken = frag_hash_to_entry[frag_hash]["max_broken"]

                    for out_dict in out_dicts:
                        out_hash = out_dict["new_hash"]
                        out_frag = out_dict["new_frag"]
                        rm_bond_t = out_dict["rm_bond_t"]
                        frag_to_hash[out_frag] = out_hash
                        current_entry = frag_hash_to_entry.get(out_hash)

                        max_broken = parent_broken + rm_bond_t

                        # Define probability of generating
                        if current_entry is None:
                            score = engine.score_fragment(int(out_frag))[1]

                            new_stack.append(out_frag)
                            new_entry = {
                                "frag": int(out_frag),
                                "frag_hash": out_hash,
                                "score": score,
                                "id": id_,
                                "parents": [frag_hash],
                                "atoms_pulled": [],
                                "left_pred": [],
                                "max_broken": max_broken,
                                "tree_depth": depth,
                                "prob_gen": prob_gen,
                            }
                            id_ += 1
                            new_entry.update(
                                engine.atom_pass_stats(out_frag, depth=max_broken)
                            )

                            # reset to best score
                            temp_form = new_entry["form"]
                            prev_best_score = form_to_min_score.get(
                                temp_form, float("inf")
                            )
                            form_to_min_score[temp_form] = min(
                                new_entry["score"], prev_best_score
                            )
                            frag_hash_to_entry[out_hash] = new_entry

                        else:
                            current_entry["parents"].append(frag_hash)
                            current_entry["prob_gen"] = max(
                                current_entry["prob_gen"], prob_gen
                            )

                        # Update cur probs
                        # This is inefficeint and can be made smarter withotu
                        # doing another minimum calculation
                        cur_probs = sorted(
                            [i["prob_gen"] for i in frag_hash_to_entry.values()]
                        )[::-1]
                        if max_nodes is None or len(cur_probs) < max_nodes:
                            min_prob = threshold
                        elif max_nodes is not None and len(cur_probs) >= max_nodes:
                            min_prob = cur_probs[max_nodes - 1]
                        else:
                            raise NotImplementedError()

                # Truncate stack; this should be handled above by min prob
                # if max_nodes is not None:
                #    new_stack = sorted(new_stack,
                #                       key=lambda x:
                #                       -frag_hash_to_entry[frag_to_hash[x]]['prob_gen'])
                #    new_stack = new_stack[:max_nodes]
                stack = new_stack

        # Only get min score for ech formula
        frag_hash_to_entry = {
            k: v
            for k, v in frag_hash_to_entry.items()
            if form_to_min_score[v["form"]] == v["score"]
        }

        if max_nodes is not None:
            sorted_keys = sorted(
                list(frag_hash_to_entry.keys()),
                key=lambda x: -frag_hash_to_entry[x]["prob_gen"],
            )
            frag_hash_to_entry = {
                k: frag_hash_to_entry[k] for k in sorted_keys[:max_nodes]
            }
        return frag_hash_to_entry
