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
Routines for handling the properties of molecules 
"""
import numpy as np
import os

from drift_composition.atoms import molecule_mass, atoms_in_molecule
from drift_composition.constants import m_hydrogen, k_boltzmann

class Molecule:
    """Wrapper for molecular properties
    
    Parameters
    ----------
    name : string
        Molecular formulation for the molecule, e.g. CO
    T_bind : float, unit=K
        Binding energy represented in K
    nu_des : float, optional. unit=s^-1
        Desorption frequency parameter. If not provided then an esimate will be
        used instead.
    """
    def __init__(self, name, T_bind, nu_des=None, ref=None):
        self._name = name
        self._mass_amu = molecule_mass(name)
        self._nu_des = nu_des
        self._T_bind = T_bind

        if nu_des is None:
            self.use_nu_des_approx()

        self._ref = ref

    def get_atoms(self):
        return atoms_in_molecule(self._name)

    @property
    def name(self):
        """Name of the molecule (chemical formula)"""
        return self._name
    
    @property
    def mass_amu(self):
        """Mass in atomic mass units"""
        return self._mass_amu
    
    @property
    def mass(self):
        """Mass in atomic mass units"""
        return self._mass_amu*m_hydrogen
    
    @property 
    def nu(self):
        """Desorbtion frequency parameter"""
        return self._nu_des
    
    @property 
    def T_bind(self):
        """Binding Energy in K"""
        return self._T_bind
    
    @property 
    def reference(self):
        """Refernce for data"""
        return self._ref
    
    def use_nu_des_approx(self, N_bind=1e15):
        """Harmonic oscillator approximation from Hasegawa et al. (1992)"""
        self._nu_des =  np.sqrt(2*N_bind*k_boltzmann*self.T_bind/(np.pi**2 * self.mass))


def get_molecular_properties(data_file=None):
    """Load the properties of molecules from a data file
    
    Parameters
    ----------
    data_file : string=default=None
        File to load the data from. Defaults to the Oberg & Wordsworth (2019)
        data.

    Returns
    -------
    molecules : list of Molecule
        Properties of the molecules
    abundances : list of abundances
        Number abundance relative to hydrogen for the molecules.
    """
    if data_file is None:
        data_file = os.path.join(
            os.path.dirname(__file__), 'chem_props_OW19.txt' 
        )

    data = np.genfromtxt(data_file, dtype=('S10', 'f8', 'f8', 'f8', 'S14'))
    molecules, abundance = [], []
    for line in data:
        nu_des = line[2] 
        if nu_des <= 0 or np.isnan(nu_des):
            nu_des = None

        molecules.append(
            Molecule(line[0].decode("ascii"),  line[3], nu_des, line[4].decode("ascii"))
        )
        abundance.append(line[1])
    
    return molecules, abundance
