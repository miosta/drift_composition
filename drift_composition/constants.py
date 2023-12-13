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
Physical Constants
"""
c_light = 2.99792458e10
h_planck = 6.62607015e-27
c2_rad = 1.4387768775039339e0  # h*c/kB in cm K
stefan_boltzmann = 5.670374419184429453970e-5

G_Msun = 1.32712442099e26
G_Newton = 6.67430e-8
Msun = G_Msun / G_Newton
Rsun = 6.96340e10
Lsun = 3.846e33

m_hydrogen = 1.6735575e-24
k_boltzmann = 1.380649e-16

Rau = 1.495978707e13

yr = 3600 * 24 * 365.25
