""" fragmentation_engine.

Fragment a molecule combinatorially by pulling atoms

"""
import numpy as np
from collections import Counter
from hashlib import blake2b
from typing import List
from rdkit import Chem
from ms_pred import common


TYPEW = {
    Chem.rdchem.BondType.names["AROMATIC"]: 3,
    Chem.rdchem.BondType.names["DOUBLE"]: 2,
    Chem.rdchem.BondType.names["TRIPLE"]: 3,
    Chem.rdchem.BondType.names["SINGLE"]: 1,
}
MAX_BONDS = max(list(TYPEW.values())) + 1
MAX_ATOM_BONDS = 6

# CC bonds are strongest --> higher score = harder to break
HETEROW = {False: 2, True: 1}


class FragmentEngine(object):
    """FragmentEngine."""

    def __init__(
        self, mol_str: str, max_broken_bonds: int = 3, mol_str_type: str = "smiles"
    ):
        """__init__.

        Args:
            mol_str (str): smiles or inchi
            max_broken_bonds (int): max_broken_bonds
            mol_str_type (str): Define smiles
        """

        # Standardize mol by roundtripping it
        if mol_str_type == "smiles":
            self.smiles = mol_str
            self.mol = Chem.MolFromSmiles(self.smiles)
            if self.mol is None:
                return
            self.inchi = Chem.MolToInchi(self.mol)
            self.mol = Chem.MolFromInchi(self.inchi)

        elif mol_str_type == "inchi":
            self.inchi = mol_str
            self.mol = Chem.MolFromInchi(self.inchi)
            if self.mol is None:
                return
            self.smiles = Chem.MolToSmiles(self.mol)
        else:
            raise NotImplementedError()

        self.natoms = self.mol.GetNumAtoms()

        # Kekulize the molecule
        Chem.Kekulize(self.mol, clearAromaticFlags=True)

        # Calculate number of hs on each atom and masses
        self.atom_symbols = [i.GetSymbol() for i in self.mol.GetAtoms()]
        self.atom_hs = np.array(
            [i.GetNumImplicitHs() + i.GetNumExplicitHs() for i in self.mol.GetAtoms()]
        )
        self.total_hs = self.atom_hs.sum()
        self.atom_weights = np.array(
            [common.ELEMENT_TO_MASS[i] for i in self.atom_symbols]
        )
        self.atom_weights_h = (
            self.atom_hs * common.ELEMENT_TO_MASS["H"] + self.atom_weights
        )

        # Get bonds and indices
        # Note: Unlike MAGMa original, we do not score bonds
        self.bonded_atoms = [[] for _ in self.atom_symbols]
        self.bonded_types = [[] for _ in self.atom_symbols]

        # For numpy use in numba algo
        self.bonded_atoms_np = np.zeros((self.natoms, MAX_ATOM_BONDS), dtype=int)
        self.bonded_types_np = np.zeros((self.natoms, MAX_ATOM_BONDS), dtype=int)
        self.num_bonds_np = np.zeros(self.natoms, dtype=int)

        self.bond_to_type = {}
        self.bonds = set()
        self.bonds_list = []
        self.bond_types_list = []
        self.bond_inds_list = []
        self.bondscore = {}
        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.bonded_atoms[a1].append(a2)
            self.bonded_atoms[a2].append(a1)

            self.bonded_atoms_np[a1, self.num_bonds_np[a1]] = a2
            self.bonded_atoms_np[a2, self.num_bonds_np[a2]] = a1

            bondbits = 1 << a1 | 1 << a2
            bondscore = (
                TYPEW[bond.GetBondType()]
                * HETEROW[self.atom_symbols[a1] != "C" or self.atom_symbols[a2] != "C"]
            )
            bondtype = TYPEW[bond.GetBondType()]

            self.bonded_types[a1].append(bondtype)
            self.bonded_types[a2].append(bondtype)

            self.bonded_types_np[a1, self.num_bonds_np[a1]] = bondtype
            self.bonded_types_np[a2, self.num_bonds_np[a2]] = bondtype

            self.num_bonds_np[a1] += 1
            self.num_bonds_np[a2] += 1

            self.bond_to_type[bondbits] = bondtype
            self.bondscore[bondbits] = bondscore
            if bondbits not in self.bonds:
                self.bonds_list.append(bondbits)
                self.bond_types_list.append(bondtype)
                self.bond_inds_list.append((a1, a2))
            self.bonds.add(bondbits)

        # Get engine params
        self.max_broken_bonds = max_broken_bonds

        # Define a list for all possible mass shifts on each atom accounting
        # for masses
        # Shift factor for inverse
        # int((self.shift_buckets.shape[0] - 1) / 2)
        self.shift_buckets = (
            np.arange(self.max_broken_bonds * 2 + 1) - self.max_broken_bonds
        )
        self.shift_bucket_inds = np.arange(self.max_broken_bonds * 2 + 1)
        self.shift_bucket_masses = self.shift_buckets * common.ELEMENT_TO_MASS["H"]

        # Define exports
        self.frag_to_entry = {}

    def atom_pass_stats(self, frag: int, depth: int = None):
        """atom_pass_stats.

        Combine logic of formula_from_frag and single_mass to avoid double
        looping over atoms

        """
        fragment_mass = 0.0
        form_vec = np.zeros(len(common.VALID_ELEMENTS))
        h_pos = common.element_to_ind["H"]
        for atom in range(self.natoms):
            if frag & (1 << atom):
                fragment_mass += self.atom_weights_h[atom]
                dense_pos = common.element_to_ind[self.atom_symbols[atom]]
                form_vec[dense_pos] += 1
                form_vec[h_pos] += self.atom_hs[atom]
        form = common.vec_to_formula(form_vec)
        frag_hs = int(form_vec[h_pos])
        max_remove = frag_hs
        max_add = self.total_hs - frag_hs

        if depth is not None:
            max_remove = int(min(depth, max_remove))
            max_add = int(min(depth, max_add))

        return {
            "form": form,
            "base_mass": float(fragment_mass),
            "max_remove_hs": max_remove,
            "max_add_hs": max_add,
        }

    def wl_hash(
        self,
        template_fragment: int,
    ) -> int:
        """wl_hash.

        Produces int hash of template fragment

        Args:
            template_fragment: Int defining template fragment

        Return:
            new_fragment
        """
        # Step 1: Define initial hashes by symbol + num hydrogens + num bonds
        cur_hashes = [
            f"{i}_{j}_{len(k)}"
            for i, j, k in zip(self.atom_symbols, self.atom_hs, self.bonded_atoms)
        ]

        cur_hashes = [
            f"{i}"
            for i, j, k in zip(self.atom_symbols, self.atom_hs, self.bonded_atoms)
        ]

        def get_graph_hash(full_hashes):
            """get_graph_hash."""
            counter = Counter(full_hashes)
            counter_str = str(tuple(sorted(counter.items(), key=lambda x: x[0])))
            counter_hash = _hash_label(counter_str)
            return counter_hash

        graph_hash = get_graph_hash(cur_hashes)
        iterations = self.natoms
        changed = True
        ct = 0

        while ct <= iterations and changed:
            new_hashes = []
            temp_atoms = 0
            # Step 2: Update hashes with local neighborhoods
            for atom in range(self.natoms):

                atombit = 1 << atom
                cur_hash = cur_hashes[atom]

                if not atombit & template_fragment:
                    new_hashes.append(cur_hash)
                    continue

                # Count num atoms in this loop
                temp_atoms += 1

                # Get local neighbors
                neighbor_labels = []
                for targind in self.bonded_atoms[atom]:
                    targbit = 1 << targind

                    if not targbit & template_fragment:
                        continue

                    targhash = cur_hashes[targind]
                    bondbit = targbit | atombit
                    bondtype = self.bond_to_type[bondbit]
                    neighbor_label = f"{bondtype}_{targhash}"
                    neighbor_labels.append(neighbor_label)
                new_hash_str = cur_hash + "".join(sorted(neighbor_labels))
                new_hash = _hash_label(new_hash_str)
                new_hashes.append(new_hash)

            # Update num atoms to be correct for fragment
            iterations = temp_atoms
            # Check if the overall hash changed
            new_graph_hash = get_graph_hash(new_hashes)
            changed = new_graph_hash != graph_hash

            # Update count and set new graph hash & cur_hash
            graph_hash = new_graph_hash
            cur_hash = new_hashes
            ct += 1
        return graph_hash

    def get_frag_forms(self):
        """get_frag_forms."""

        masses, form_vecs = [], []

        form_set = set()

        for k, v in self.frag_to_entry.items():
            max_remove, max_add = v["max_remove_hs"], v["max_add_hs"]
            base_mass = v["base_mass"]
            base_form_str = v["form"]
            base_form_vec = common.formula_to_dense(base_form_str)

            for num_shift, shift_ind, shift_mass in zip(
                self.shift_buckets, self.shift_bucket_inds, self.shift_bucket_masses
            ):
                if (num_shift >= -max_remove) and (num_shift <= max_add):
                    new_form_vec = (
                        base_form_vec
                        + num_shift * common.ELEMENT_VECTORS[common.element_to_ind["H"]]
                    )
                    str_code = str(new_form_vec)
                    if str_code in form_set:
                        continue

                    # Add to output
                    masses.append(base_mass + shift_mass)
                    form_vecs.append(new_form_vec)

                    # Update Set of included
                    form_set.add(str_code)

        return np.array(form_vecs), np.array(masses)

    def generate_fragments(self):
        """generate_fragments.

        Pull atoms sequentially from molecule to create fragments. Populates
        self.frag_to_entry.
        """

        # This is a clever way to assert that all atoms are present; everything
        # will be kept as bit strings i.e., take 1 and move it all the way to
        # the left in binary notation byself.natoms positions and subtract 1;
        # that means every atom is present (all 1's in binary format)
        cur_id = 0
        frag = (1 << self.natoms) - 1
        root = {
            "frag": frag,
            "id": cur_id,
            "max_broken": 0,
        }
        frag_hash = self.wl_hash(frag)
        root.update(self.atom_pass_stats(frag, depth=0))

        self.frag_to_entry[frag_hash] = root
        current_fragments = [frag_hash]
        new_fragments = []

        # generate fragments for max_broken_bond steps
        for step in range(self.max_broken_bonds):

            # loop over all fragments to be fragmented
            for frag_hash in current_fragments:

                # Get current parent node we are sub fragmenting
                fragment = self.frag_to_entry[frag_hash]["frag"]
                # loop over all atoms in the fragment
                for atom in range(self.natoms):
                    extended_fragments = self.remove_atom(fragment, atom)
                    for frag in extended_fragments:

                        # unpack
                        new_frag_hash = frag["new_hash"]
                        frag = frag["new_frag"]

                        # add extended fragments, if not yet present, to the collection
                        old_entry = self.frag_to_entry.get(new_frag_hash)
                        max_broken = step + 1
                        if old_entry is None:
                            cur_id += 1
                            new_entry = {
                                "frag": frag,
                                "id": cur_id,
                                "max_broken": max_broken,
                            }
                            new_entry
                            new_entry.update(
                                self.atom_pass_stats(frag, depth=max_broken)
                            )
                            self.frag_to_entry[new_frag_hash] = new_entry
                            new_fragments.append(new_frag_hash)
            current_fragments = new_fragments
            new_fragments = []

    def remove_atom(self, fragment: int, atom: int) -> List[dict]:
        """remove_atom.

        Remove an atom from the fragment and get all resulting frag int dicts

        Args:
            fragment (int): fragment
            atom (int): atom

        Returns:
            List[dict]: All new fragments
        """

        # Skip if atom is not in the current fragment
        if not ((1 << atom) & fragment):
            return []

        #  Compute the template fragment after removing this atom
        template_fragment = fragment ^ (1 << atom)
        list_ext_atoms = set([])
        extended_fragments = []

        # Get all neighboring atoms to the removed atom
        for a in self.bonded_atoms[atom]:

            # present in the fragment
            if (1 << a) & template_fragment:
                list_ext_atoms.add(a)

        # In case of one bonded atom, the new fragment is the remainder of the old fragment
        if len(list_ext_atoms) == 1:
            if template_fragment == 0:
                return []

            new_frag_hash = self.wl_hash(template_fragment)
            extended_fragments.append(
                {
                    "new_frag": template_fragment,
                    "new_hash": new_frag_hash,
                    "removed_atom": atom,
                }
            )
        else:
            # otherwise extend each neighbor atom to a complete fragment
            for a in list_ext_atoms:
                # except when deleted atom is in a ring and a previous extended
                # fragment already contains this neighbor atom, then
                # calculate fragment only once
                is_ring = False
                for frag_dict in extended_fragments:
                    frag_id = frag_dict["new_frag"]
                    if (1 << a) & frag_id:
                        is_ring = True
                if not is_ring:
                    # extend atom to complete fragment
                    new_fragment = extend(a, self.bonded_atoms, template_fragment)
                    if new_fragment == 0:
                        continue

                    new_frag_hash = self.wl_hash(new_fragment)
                    extended_fragments.append(
                        {
                            "new_frag": new_fragment,
                            "new_hash": new_frag_hash,
                            "removed_atom": atom,
                        }
                    )
        return extended_fragments


def extend(
    atom: int,
    bonded_atoms: list,
    template_fragment: int,
):
    """extend.

    Dfs extension of atom to other parts of the template fragment

    Args:
        atom (int): Int representing position of atom to extend
        bonded_atoms (List[List[int]]): List mapping current atoms to other
            atoms
        template_fragment (int): Int representing binary of all atoms currently
            in the fragment
    Return:
        new_fragment
    """
    stack = [atom]
    new_fragment = 0
    while len(stack) > 0:
        atom = stack.pop()
        for a in bonded_atoms[atom]:
            atombit = 1 << a
            # If new atom is not in the template fragment
            if (not (atombit & template_fragment)) or (atombit & new_fragment):
                continue

            # If its in template fragment and not yet in new fragment
            new_fragment = new_fragment | atombit
            stack.append(a)
    return new_fragment


def _hash_label(label, digest_size=32):
    """_hash_label"""
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def bit_array(num):
    """Convert a positive integer num into a bit vector"""
    return np.array(list(f"{num:b}")).astype(float)
