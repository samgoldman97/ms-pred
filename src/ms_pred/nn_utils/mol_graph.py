""" mol_graph.py.

Classes to featurize molecules into a graph with onehot concat feats on atoms
and bonds. Inspired by the dgllife library.

"""
from rdkit import Chem
import numpy as np
import torch
import dgl
import ms_pred.nn_utils.nn_utils as nn_utils

atom_feat_registry = {}
bond_feat_registry = {}


def register_bond_feat(cls):
    """register_bond_feat."""
    bond_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


def register_atom_feat(cls):
    """register_atom_feat."""
    atom_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


class MolDGLGraph:
    def __init__(
        self,
        atom_feats: list = [
            "a_onehot",
            "a_degree",
            "a_hybrid",
            "a_formal",
            "a_radical",
            "a_ring",
            "a_mass",
            "a_chiral",
        ],
        bond_feats: list = ["b_degree"],
        pe_embed_k: int = 0,
    ):
        """__init__

        Args:
            atom_feats (list)
            bond_feats (list)
            pe_embed_k (int)

        """
        self.pe_embed_k = pe_embed_k
        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.a_featurizers = []
        self.b_featurizers = []

        self.num_atom_feats = 0
        self.num_bond_feats = 0

        for i in self.atom_feats:
            if i not in atom_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = atom_feat_registry[i]
            self.num_atom_feats += feat_obj["feat_size"]
            self.a_featurizers.append(feat_obj["fn"])

        for i in self.bond_feats:
            if i not in bond_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = bond_feat_registry[i]
            self.num_bond_feats += feat_obj["feat_size"]
            self.b_featurizers.append(feat_obj["fn"])

        self.num_atom_feats += self.pe_embed_k

    def get_mol_graph(
        self,
        mol: Chem.Mol,
        bigraph: str = True,
    ) -> dict:
        """get_mol_graph.

        Args:
            mol (Chem.Mol):
            bigraph (bool): If true, double all edges.

        Return:
            dict:
                "atom_feats": np.ndarray (|N| x d_n)
                "bond_feats": np.ndarray (|E| x d_e)
                "bond_tuples": np.ndarray (|E| x 2)

        """
        all_atoms = mol.GetAtoms()
        all_bonds = mol.GetBonds()
        bond_feats = []
        bond_tuples = []
        atom_feats = []
        for bond in all_bonds:
            strt = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_tuples.append((strt, end))
            bond_feat = []
            for fn in self.b_featurizers:
                bond_feat.extend(fn(bond))
            bond_feats.append(bond_feat)

        for atom in all_atoms:
            atom_feat = []
            for fn in self.a_featurizers:
                atom_feat.extend(fn(atom))
            atom_feats.append(atom_feat)

        atom_feats = np.array(atom_feats)
        bond_feats = np.array(bond_feats)
        bond_tuples = np.array(bond_tuples)

        # Add doubles
        if bigraph:
            rev_bonds = np.vstack([bond_tuples[:, 1], bond_tuples[:, 0]]).transpose(
                1, 0
            )
            bond_tuples = np.vstack([bond_tuples, rev_bonds])
            bond_feats = np.vstack([bond_feats, bond_feats])
        return {
            "atom_feats": atom_feats,
            "bond_feats": bond_feats,
            "bond_tuples": bond_tuples,
        }

    def get_dgl_graph(self, mol: Chem.Mol, bigraph: str = True):
        """get_dgl_graph.

        Args:
            mol (Chem.Mol):
            bigraph (bool): If true, double all edges.

        Return:
            dgl graph object
        """
        mol_graph = self.get_mol_graph(mol, bigraph=bigraph)

        bond_inds = torch.from_numpy(mol_graph["bond_tuples"]).long()
        bond_feats = torch.from_numpy(mol_graph["bond_feats"]).float()
        atom_feats = torch.from_numpy(mol_graph["atom_feats"]).float()

        g = dgl.graph(
            data=(bond_inds[:, 0], bond_inds[:, 1]), num_nodes=atom_feats.shape[0]
        )
        g.ndata["h"] = atom_feats
        g.edata["e"] = bond_feats

        if self.pe_embed_k > 0:
            pe_embeds = nn_utils.random_walk_pe(
                g,
                k=self.pe_embed_k,
            )
            g.ndata["h"] = torch.cat((g.ndata["h"], pe_embeds), -1)

        return g


class FeatBase:
    """FeatBase.

    Extend this class for atom and bond featurizers

    """

    feat_size = 0
    name = "base"

    @classmethod
    def featurize(cls, x) -> list:
        raise NotImplementedError()


@register_atom_feat
class AtomOneHot(FeatBase):
    name = "a_onehot"
    allowable_set = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "H",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetSymbol(), cls.allowable_set, True)


@register_atom_feat
class AtomDegree(FeatBase):
    name = "a_degree"
    allowable_set = list(range(11))
    feat_size = len(allowable_set) + 1 + 2

    @classmethod
    def featurize(cls, x) -> int:
        deg = [x.GetDegree(), x.GetTotalDegree()]
        onehot = one_hot_encoding(deg, cls.allowable_set, True)
        return deg + onehot


@register_atom_feat
class AtomHybrid(FeatBase):

    name = "a_hybrid"
    allowable_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        onehot = one_hot_encoding(x.GetHybridization(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomFormal(FeatBase):

    name = "a_formal"
    allowable_set = list(range(-2, 3))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        onehot = one_hot_encoding(x.GetFormalCharge(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRadical(FeatBase):

    name = "a_radical"
    allowable_set = list(range(5))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        onehot = one_hot_encoding(x.GetNumRadicalElectrons(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRing(FeatBase):

    name = "a_ring"
    allowable_set = [True, False]
    feat_size = len(allowable_set) * 2

    @classmethod
    def featurize(cls, x) -> int:
        onehot_ring = one_hot_encoding(x.IsInRing(), cls.allowable_set, False)
        onehot_aromatic = one_hot_encoding(x.GetIsAromatic(), cls.allowable_set, False)
        return onehot_ring + onehot_aromatic


@register_atom_feat
class AtomChiral(FeatBase):

    name = "a_chiral"
    allowable_set = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        chiral_onehot = one_hot_encoding(x.GetChiralTag(), cls.allowable_set, True)
        return chiral_onehot


@register_atom_feat
class AtomMass(FeatBase):

    name = "a_mass"
    coef = 0.01
    feat_size = 1

    @classmethod
    def featurize(cls, x) -> int:
        return [x.GetMass() * cls.coef]


@register_bond_feat
class BondDegree(FeatBase):

    name = "b_degree"
    allowable_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetBondType(), cls.allowable_set, True)


@register_bond_feat
class BondStereo(FeatBase):

    name = "b_stereo"
    allowable_set = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetStereo(), cls.allowable_set, True)


@register_bond_feat
class BondConj(FeatBase):

    name = "b_ring"
    feat_size = 2

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.IsInRing(), [False, True], False)


@register_bond_feat
class BondConj(FeatBase):

    name = "b_conj"
    feat_size = 2

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetIsConjugated(), [False, True], False)


def one_hot_encoding(x, allowable_set, encode_unknown=False) -> list:
    """One_hot encoding.

    Code taken from dgllife library
    https://lifesci.dgl.ai/_modules/dgllife/utils/featurizers.html

    Args:
        x: Val to encode
        allowable_set: Options
        encode_unknown: If true, encode unk

    Return:
        list of bools
    """

    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: int(x == s), allowable_set))
