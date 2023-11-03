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
Grid structure for the disc
"""
import numpy as np
from drift_composition.reconstruct import compute_centroids


class Grid:
    """
    Basic grid structure for a disc.

    Parameters
    ----------
    R_in : float
        Inner radius of the disc model
    R_out : float
        Outer radius of the disc model
    N : integer
        Number of cells to use
    """

    def __init__(self, R_in, R_out, N):
        self.Re = np.geomspace(R_in, R_out, N + 1)
        self.Rc = compute_centroids(self.Re, self.order)

    @property
    def cell_vol(self):
        """Volume of the cells"""
        return np.pi * np.diff(self.Re**2)

    @property
    def face_area(self):
        """Area of the cell faces"""
        return 2 * np.pi * self.Re

    @property
    def size(self):
        """Number of cells"""
        return len(self.Rc)

    @property
    def order(self):
        """Order of the volume element: dV ~ R^order dR"""
        return 1
