""" joint_model. """
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.dag_pred.gen_model as gen_model
import ms_pred.dag_pred.inten_model as inten_model
import ms_pred.dag_pred.dag_data as dag_data


class JointModel(pl.LightningModule):
    def __init__(
        self,
        gen_model_obj: gen_model.FragGNN,
        inten_model_obj: inten_model.IntenGNN,
    ):
        """__init__.

        Args:
            gen_model_obj (gen_model.FragGNN): gen_model_obj
            inten_model_obj (inten_model.IntenGNN): inten_model_obj
        """

        super().__init__()
        self.gen_model_obj = gen_model_obj
        self.inten_model_obj = inten_model_obj
        self.inten_collate_fn = dag_data.IntenPredDataset.get_collate_fn()

        root_enc_gen = self.gen_model_obj.root_encode
        pe_embed_gen = self.gen_model_obj.pe_embed_k
        add_hs_gen = self.gen_model_obj.add_hs

        root_enc_inten = self.inten_model_obj.root_encode
        pe_embed_inten = self.inten_model_obj.pe_embed_k
        add_hs_inten = self.inten_model_obj.add_hs

        self.gen_tp = dag_data.TreeProcessor(
            root_encode=root_enc_gen, pe_embed_k=pe_embed_gen, add_hs=add_hs_gen
        )

        self.inten_tp = dag_data.TreeProcessor(
            root_encode=root_enc_inten, pe_embed_k=pe_embed_inten, add_hs=add_hs_inten
        )

    @classmethod
    def from_checkpoints(cls, gen_checkpoint, inten_checkpoint):
        """from_checkpoints.

        Args:
            gen_checkpoint
            inten_checkpoint
        """

        gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_checkpoint)
        inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_checkpoint)
        return cls(gen_model_obj, inten_model_obj)

    def predict_mol(
        self,
        smi: str,
        adduct: str,
        threshold: float,
        device: str,
        max_nodes: int,
        binned_out: bool = False,
    ):
        """predict_mol.

        Args:
            smi (str): smi
            adduct
            threshold (float): threshold
            device (str): device
            max_nodes (int): max_nodes
            binned_out
        """

        self.eval()
        self.freeze()

        # Run tree gen model
        # Defines exact tree
        root_smi = smi
        root_inchi = common.inchi_from_smiles(root_smi)

        frag_tree = self.gen_model_obj.predict_mol(
            root_smi=root_smi,
            adduct=adduct,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
        )
        frag_tree = {"root_inchi": root_inchi, "name": "", "frags": frag_tree}
        
        # Get engine from fragmentation for this inchi
        engine = fragmentation.FragmentEngine(mol_str=root_inchi,
                                              mol_str_type="inchi"
        )

        processed_tree = self.inten_tp.process_tree_inten_pred(frag_tree)

        # Save for output wrangle
        out_tree = processed_tree["tree"]
        processed_tree = processed_tree["dgl_tree"]

        processed_tree["adduct"] = common.ion2onehot_pos[adduct]
        processed_tree["name"] = ""
        batch = self.inten_collate_fn([processed_tree])
        inten_frag_ids = batch["inten_frag_ids"]

        safe_device = lambda x: x.to(device) if x is not None else x

        frag_graphs = safe_device(batch["frag_graphs"])
        root_reprs = safe_device(batch["root_reprs"])
        ind_maps = safe_device(batch["inds"])
        num_frags = safe_device(batch["num_frags"])
        broken_bonds = safe_device(batch["broken_bonds"])
        max_remove_hs = safe_device(batch["max_remove_hs"])
        max_add_hs = safe_device(batch["max_add_hs"])
        masses = safe_device(batch["masses"])

        adducts = safe_device(batch["adducts"]).to(device)
        root_forms = safe_device(batch["root_form_vecs"])
        frag_forms = safe_device(batch["frag_form_vecs"])

        # IDs to use to recapitulate
        inten_preds = self.inten_model_obj.predict(
            graphs=frag_graphs,
            root_reprs=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            max_breaks=broken_bonds,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
            binned_out=binned_out,
            adducts=adducts,
        )

        if binned_out:
            out = inten_preds
        else:
            inten_preds = inten_preds["spec"][0]
            inten_frag_ids = inten_frag_ids[0]
            out_frags = out_tree["frags"]

            # Get masses too 
            for inten_pred, inten_frag_id in zip(inten_preds, inten_frag_ids):
                out_frags[inten_frag_id]["intens"] = inten_pred.tolist()

                new_masses = out_frags[inten_frag_id]["base_mass"] + engine.shift_bucket_masses
                mz_with_charge = new_masses + common.ion2mass[adduct]
                out_frags[inten_frag_id]["mz_no_charge"] = new_masses.tolist()
                out_frags[inten_frag_id]["mz_charge"] = mz_with_charge.tolist()

            out_tree["frags"] = out_frags
            out = out_tree
        return out
