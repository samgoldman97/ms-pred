"""chem_utils.py"""

import re
import numpy as np
import pandas as pd
from functools import reduce

import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "Na",
    "K",
]


# Set the exact molecular weight?
# Use this to define an element priority queue
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]
CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)


ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))

# Convert to onehot
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

# Use Monoisotopic
# VALID_MASSES = np.array([Atom(i).GetMass() for i in VALID_ELEMENTS])
VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC_MASS = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1, 1471]
)

NORM_VEC = np.array([81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1])
MAX_ATOM_CT = 160


# For helping binary conversions
BINARY_BITS = 8

# Assume 64 is the highest repeat of any 1 atom
MAX_ELEMENT_NUM = 64

# Hydrogen featurizer
MAX_H = 6

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))

# Map ion to adduct mass, don't use electron
ion2mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] + ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-H2O]+": -ELEMENT_TO_MASS["O"] + ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+NH4]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M+H-2H2O]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
}

# Valid adducts
ion2onehot_pos = {
    "[M+H]+": 0,
    "[M+Na]+": 1,
    "[M+K]+": 2,
    "[M-H2O+H]+": 3,
    "[M+H-H2O]+": 3,
    "[M+H3N+H]+": 4,
    "[M+NH4]+": 4,
    "[M]+": 5,
    "[M-H4O2+H]+": 6,
    "[M+H-2H2O]+": 6,
}


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def cross_sum(x, y):
    """cross_sum."""
    return (x[None, :, :] + y[:, None, :]).reshape(-1, y.shape[-1])


def get_all_subsets(chem_formula: str) -> (np.ndarray, np.ndarray):
    """get_all_subsets.

    Args:
        chem_formula (str): Chem formula
    Return:
        Tuple of vecs and their masses
    """

    dense_formula = formula_to_dense(chem_formula)
    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = [
        ELEMENT_VECTORS[nonzero_ind]
        * np.arange(0, dense_formula[nonzero_ind] + 1)[:, None]
        for nonzero_ind in non_zero
    ]

    cross_prod = reduce(cross_sum, vectorized_formula)
    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]

    all_masses = np.einsum("ij,j->i", cross_prod, VALID_MONO_MASSES)
    return cross_prod, all_masses


def rdbe_filter(cross_prod):
    """rdbe_filter.

    Args:
        cross_prod:
    """
    # Filter
    pos_els = ["C", "C", "N", "P"]
    neg_els = ["H", "Cl", "Br", "I", "F"]

    # Apply rdbe filter
    # RDBE = 1 + 0.5 * (2#C − #H +#N+#P−#Cl−#Br−#I−#F)
    rdbe_total = np.zeros(cross_prod.shape[0])
    for pos_el in pos_els:
        rdbe_total += cross_prod[:, element_to_ind[pos_el]]

    for neg_el in neg_els:
        rdbe_total -= cross_prod[:, element_to_ind[neg_el]]

    # Manage
    rdbe_total = 1 + 0.5 * rdbe_total
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds


def formula_to_dense_mass(chem_formula: str) -> np.ndarray:
    """formula_to_dense_mass.

    Return formula including full compound mass

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position_mass[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position_mass["H"]))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_dense_mass_norm(chem_formula: str) -> np.ndarray:
    """formula_to_dense_mass_norm.

    Return formula including full compound mass and normalized

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    dense_vec = formula_to_dense_mass(chem_formula)
    dense_vec = dense_vec / NORM_VEC_MASS

    return dense_vec


def formula_mass(chem_formula: str) -> float:
    """get formula mass"""
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def formula_difference(formula_1, formula_2):
    """formula_1 - formula_2"""
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }

    for k, v in form_2.items():
        form_1[k] = form_1[k] - form_2[k]

    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v > 0])
    return out_formula


def standardize_form(i):
    return vec_to_formula(formula_to_dense(i))


def get_mol_from_structure_string(structure_string, structure_type):
    if structure_type == "InChI":
        mol = Chem.MolFromInchi(structure_string)
    else:
        mol = Chem.MolFromSmiles(structure_string)
    return mol


def vec_to_formula(form_vec):
    """vec_to_formula."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def calc_structure_string_type(structure_string):
    """calc_structure_string_type.

    Args:
        structure_string:
    """
    structure_type = None
    if pd.isna(structure_string):
        structure_type = "empty"
    elif structure_string.startswith("InChI="):
        structure_type = "InChI"
    elif Chem.MolFromSmiles(structure_string) is not None:
        structure_type = "Smiles"
    return structure_type


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "inchi":
        mol = Chem.MolFromInchi(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def form_from_smi(smi: str) -> str:
    """form_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    return uncharged_formula(smi, mol_type="smiles")


def form_from_inchi(inchi: str) -> str:
    """form_from_inchi.

    Args:
        inchi (str): inchi

    Return:
        str
    """
    return uncharged_formula(inchi, mol_type="inchi")
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return ""
    else:
        return CalcMolFormula(mol)


def inchikey_from_smiles(smi: str) -> str:
    """inchikey_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)


def inchi_from_smiles(smi: str) -> str:
    """inchi_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchi(mol)


def smi_inchi_round_mol(smi: str) -> Chem.Mol:
    """smi_inchi_round.

    Args:
        smi (str): smi

    Returns:
        mol:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    inchi = Chem.MolToInchi(mol)
    if inchi is None:
        return None

    mol = Chem.MolFromInchi(inchi)
    return mol


def smiles_from_inchi(inchi: str) -> str:
    """smiles_from_inchi.

    Args:
        inchi (str): inchi

    Returns:
        str:
    """
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol)


def contains_metals(formula: str) -> bool:
    """returns true if formula contains metals"""
    METAL_RE = "(Fe|Co|Zn|Rh|Pt|Li)"
    return len(re.findall(METAL_RE, formula)) > 0


class SmilesStandardizer(object):
    """Standardize smiles"""

    def __init__(self, *args, **kwargs):
        self.fragment_standardizer = rdMolStandardize.LargestFragmentChooser()
        self.charge_standardizer = rdMolStandardize.Uncharger()

    def standardize_smiles(self, smi):
        """Standardize smiles string"""
        mol = Chem.MolFromSmiles(smi)
        out_smi = self.standardize_mol(mol)
        return out_smi

    def standardize_mol(self, mol) -> str:
        """Standardize smiles string"""
        mol = self.fragment_standardizer.choose(mol)
        mol = self.charge_standardizer.uncharge(mol)

        # Round trip to and from inchi to tautomer correct
        # Also standardize tautomer in the middle
        output_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        return output_smi


def mass_from_inchi(inchi: str) -> float:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def mass_from_smi(smi: str) -> float:
    """mass_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def min_formal_from_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.min()


def max_formal_from_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.max()


def atoms_from_smi(smi: str) -> int:
    """atoms_from_smi.

    Args:
        smi (str): smi

    Return:
        int
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return mol.GetNumAtoms()


def num_to_binary(num, num_bits=BINARY_BITS):
    """num_to_binary.

    Args:
        num:
        num_bits:
    """
    format_obj = "{:0" + str(num_bits) + "b}"
    out = format_obj.format(num)
    out = np.array(list(out)).astype(int)
    return out


def has_valid_els(chem_formula: str) -> bool:
    """has_valid_els"""
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        if chem_symbol not in VALID_ELEMENTS:
            return False
    return True
