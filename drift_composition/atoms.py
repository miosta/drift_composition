# drift_composition: A 1D model for the advection and diffusion of molecules
# through protoplanetary discs.
#
# Copyright (C) 2023  R. Booth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
"""
Functions defining properties of atoms and decomposing molecules into 
their components.
"""
import os
from collections import defaultdict
import re
import numpy as np

# Mass of various elements in amu
ELEMENT_MASS = {
    "H": 1,
    "D": 2,
    "He": 4,
    "C": 12,
    "N": 14,
    "O": 16,
    "Ne": 20,
    "Na": 23,
    "Mg": 24,
    "Al": 27,
    "Si": 28,
    "P": 31,
    "S": 32,
    "Cl": 36,
    "Ar": 40,
    "K": 39,
    "Ca": 40,
    "Fe": 56,
    "Kr": 84,
    "Xe": 131,
}


def atoms_in_molecule(mol_in):
    """Get the atoms that make up a molecule"""
    atoms = list(ELEMENT_MASS.keys())

    atoms1 = [atom for atom in atoms if len(atom) == 1]
    atoms2 = [atom for atom in atoms if len(atom) == 2]

    atoms = defaultdict(int)
    mol = mol_in
    for atom_list in [atoms2, atoms1]:
        for atom in atom_list:
            while True:
                # Determine whether molecule contains this atom
                m = re.search(atom, mol)
                if m is None:
                    break

                # Determine whether there are numbers after the name
                digit = m.end()
                while digit < len(mol) and mol[digit].isdigit():
                    digit += 1

                # Get the number of times this element appears in the molecule
                # (at this location)
                if digit == m.end():
                    count = 1
                else:
                    count = int(mol[m.end() : digit])

                atoms[atom] += count

                # Remove the element from the molecule
                mol = mol[: m.start()] + mol[digit:]

    if len(mol) != 0:
        print(mol)
        raise ValueError(f"Molecule {mol_in} contains unrecognized components {mol}")

    return atoms


def molecule_mass(mol):
    """Get the mass of a molecule in amu"""
    atoms = atoms_in_molecule(mol)
    mass = 0
    for atom in atoms:
        mass += atoms[atom] * ELEMENT_MASS[atom]
    return mass


def load_protosolar_abundances(data_file=None):
    """The (proto)-solar abundances from Asplund."""
    if data_file is None:
        data_file = os.path.join(os.path.dirname(__file__), "proto_solar.dat")
    data = np.genfromtxt(data_file, dtype=("i4", "S2", "f8", "f8"))

    elements, abundances = [], []
    for item in data:
        elements.append(item[1].decode("ascii"))
        # Convert from log_10 + 12 notation to abundance.
        abundances.append(10 ** (item[2] - 12))
    abundances = np.array(abundances) / np.sum(abundances)

    return {e: a for e, a in zip(elements, abundances)}
