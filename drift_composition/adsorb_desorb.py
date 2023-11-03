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
Routines for modelling the adsorption and desorption of molecules from dust 
grains.
"""
import numpy as np

from drift_composition.constants import k_boltzmann, m_hydrogen


class GrainModel:
    """Model for the grain properties

    Parameters
    ----------
    a_min : float, default=1e-5, unit=cm
        Size of the smallest grains
    rho : float, default=1, unit=g/cm^3
        Internal density of a dust grain
    q : float, default=3.5
        Slope of the grain size density n(a)da ~ a^-q da
    N_bind : float, default=1e15, unit=cm^-2
        Number of binding sites per unit area of the grain surface

    Notes
    -----
    Assumes compact spherical grains
    """

    def __init__(self, a_min=1e-5, rho=1, q=3.5, N_bind=1e15):
        self._a0 = a_min
        self._rho = rho
        self._q = q
        self._N_bind = N_bind

    def area(self, a_max):
        """Get the average area of the grains"""
        q = self._q
        a_min = self._a0

        area = (a_max ** (3 + q) - a_min ** (3 + q)) / (3 + q)
        N = (a_max ** (1 + q) - a_min ** (1 + q)) / (1 + q)

        return np.pi * area / N

    def mass(self, a_max):
        """Get the average mass of the grains"""
        q = self._q
        a_min = self._a0

        mass = (a_max ** (4 + q) - a_min ** (4 + q)) / (4 + q)
        N = (a_max ** (1 + q) - a_min ** (1 + q)) / (1 + q)

        return (4 * np.pi * self._rho / 3) * mass / N

    @property
    def rho(self):
        """Internal density of the grain"""
        return self._rho

    @property
    def N_bind(self):
        """Number of binding sites per unit area"""
        return self._N_bind

    @property
    def a_min(self):
        """Minimum grain size"""
        return self._a0

    @property
    def q(self):
        """Slope of the grain size distribution"""
        return self._q


class AdsorptionRate:
    """Model for themal adsorption onto grains.

    Parameters
    ----------
    species : Molecule object
        Molecule we are interested in.
    grain : GrainModel
        Properties of the dust grains
    p_stick : float, default=1
        Probability that dust-gas collisions lead to sticking

    Notes
    -----
    Assumes that the dust grains are settled to the mid-plane and
    the gas is spread of the disc.
    """

    def __init__(self, species, grain, p_stick=1):
        self._species = species
        self._grain = grain
        self._p_stick = p_stick

    def __call__(self, Sigma_mol, Sigma_dust, T, H, a_max):
        """Compute the thermal adsorption rate.

        Parameters
        ----------
        Sigma_mol : float or array
            Surface density of the adsorbing molecules
        Sigma_dust : float or array
            Surface density of the dust
        T : float or array
            Gas temperature
        H : float or array
            Disc vertical scale-height
        a_max : float or array
            Maximum size of the grains

        Returns
        -------
        rate : float or array
            The adsorption rate of the molecule per unit surface density of
            the molecule
        jac : float or array
            The derivative of rate with respect to the log(Sigma_mol)
        """
        v = np.sqrt(8 * k_boltzmann * T / (np.pi * self._species.mass))

        n_grain = Sigma_dust / self._grain.mass(a_max)
        area = self._grain.area(a_max)

        rate = n_grain * v * area / (np.sqrt(2 * np.pi) * H)

        return rate, 0 * rate

    @property
    def species(self):
        return self._species

    @property
    def grain(self):
        return self._grain


class ThermalDesorbtionRate:
    """Model for themal desorption onto grains.

    Parameters
    ----------
    species : Molecule object
        Molecule we are interested in.
    grain : GrainModel
        Properties of the dust grains
    p_stick : float, default=1
        Probability that dust-gas collisions lead to sticking

    Notes
    -----
    Assumes that the dust grains are settled to the mid-plane and
    the gas is spread of the disc.
    """

    def __init__(self, species, grain):
        self._species = species
        self._grain = grain

    def __call__(self, Sigma_mol, Sigma_dust, T, H, a_max):
        """Compute the thermal desorption rate.

        Parameters
        ----------
        Sigma_mol : float or array
            Surface density of the adsorbing molecules
        Sigma_dust : float or array
            Surface density of the dust
        T : float or array
            Gas temperature
        H : float or array
            Disc vertical scale-height
        a_max : float or array
            Maximum size of the grains

        Returns
        -------
        rate : float or array
            The thermal desorption rate of the molecule per unit surface
            density of the molecule
        jac : float or array
            The derivative of rate with respect to the log(Sigma_mol)
        """
        R = self._species.nu * np.exp(-self._species.T_bind / T)

        n_grain = Sigma_dust / self._grain.mass(a_max)
        area = self._grain.area(a_max)

        mass_per_layer = n_grain * area * self._grain.N_bind * self._species.mass
        num_layers = Sigma_mol / mass_per_layer

        # Only the top layer is active
        rate = R / (1 + num_layers)
        jac = -R * num_layers / (1 + num_layers) ** 2
        return rate, jac
