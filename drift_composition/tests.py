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
Tests of core functionality
"""
import numpy as np


def test_reconstruct():
    from drift_composition.reconstruct import (
        construct_FV_edge_weights, 
        join_symmetric_stencil,
        build_sparse_matrix
    )
    
    xi = np.arange(15)
    wp, wm = construct_FV_edge_weights(xi, 0, 2)

    np.testing.assert_allclose(wp[1:-2], wm[2:-1])

    # Check that it runs reasonably quickly and that we can join symmetric
    # stencils
    xi = np.logspace(-1, 2, 3*10**2+1)
    wp, wm = construct_FV_edge_weights(xi, 1, 2)
    w =  join_symmetric_stencil(wp, wm)
    
    np.testing.assert_allclose(w[0], wm[0])
    np.testing.assert_allclose(w[1], wp[0])

    ### Run a test interpolation / derivative

    # Volume average of 1/x^2
    Vx = 0.5 * (xi[1:]**2 -  xi[:-1]**2)                               
    fx = np.log(xi[1:]/xi[:-1]) / Vx

    # Check interpolation and derivaives of 1/x^2
    stencil=np.arange(-2, 2)
    w_mat = build_sparse_matrix(w[:,0],stencil)
    np.testing.assert_allclose(w_mat.dot(fx)[2:-2], 1/xi[2:-2]**2, rtol=2e-6)
    np.testing.assert_allclose(w_mat.dot(fx)[0], 1/xi[0]**2, rtol=2e-3)
    
    w_mat = build_sparse_matrix(w[:,1],stencil)
    np.testing.assert_allclose(w_mat.dot(fx)[2:-2], -2/xi[2:-2]**3, rtol=2e-6)
    np.testing.assert_allclose(w_mat.dot(fx)[0], -2/xi[0]**3, rtol=7e-2)

    w_mat = build_sparse_matrix(w[:,2],stencil)
    np.testing.assert_allclose(w_mat.dot(fx)[2:-2], 6/xi[2:-2]**4, rtol=3e-3)

    w_mat = build_sparse_matrix(w[:,3],stencil)
    np.testing.assert_allclose(w_mat.dot(fx)[2:-2], -24/xi[2:-2]**5, rtol=3e-4)

def test_adv_diff():
    from drift_composition.grid import Grid
    from drift_composition.advect_diffuse import FV_Model

    g = Grid(1, 10, 128)
    FV = FV_Model(g, 3, fields=3)

    v1 = -0.01 * g.Re**-0.5
    v2 = -0.001 * np.ones_like(g.Re)
    D = 1e-5*g.Re

    # Create 2 indepdent fields advecting, and a 3rd
    # field without advection but with exchange terms between
    # it and the first.
    adv1 = FV.create_advection_matrix(v1, 0)
    diff1 = FV.create_diffusion_matrix(D, 0, S=1/g.Rc**0.5)

    adv2 = FV.create_advection_matrix(v2, 1)
    diff2 = FV.create_diffusion_matrix(D, 1, S=1/g.Rc)

    rate = np.ones_like(g.Rc)
    ex1 = FV.create_mass_exchange_matrix(rate, 0, 2)
    ex2 = FV.create_mass_exchange_matrix(2*rate, 2, 0)

    mat = adv1 + diff1 + adv2 + diff2 + ex1 + ex2

    # Add a source term of a constant mass accretion rate to
    # the last cell of the two advecting fields
    source = np.zeros(3*g.Rc.size, dtype='f8')
    source[-2] = source[-3] = -1

    # Solve for the surface density and compare to the analytic
    # solution
    Sigma = mat.solve(source)
    Sigma1 = Sigma[FV.get_field_ids(0)]
    Sigma2 = Sigma[FV.get_field_ids(1)]
    Sigma3 = Sigma[FV.get_field_ids(2)]

    Exact1 = 1/(2*np.pi*g.Rc*0.01*g.Rc**-0.5)
    Exact2 = 1/(2*np.pi*g.Rc*0.001)
    Exact3 = Exact1/2


    np.testing.assert_allclose(Sigma1, Exact1, rtol=1.2e-5)
    np.testing.assert_allclose(Sigma2, Exact2, rtol=3.3e-5)
    np.testing.assert_allclose(Sigma3, Exact3, rtol=1.2e-5)

    """
    import matplotlib.pyplot as plt
    plt.loglog(g.Rc, Sigma1)
    plt.loglog(g.Rc, Exact1, c='k', ls=':') 
    plt.loglog(g.Rc, Sigma2)
    plt.loglog(g.Rc, Exact2, c='k', ls=':') 
    plt.loglog(g.Rc, Sigma3)
    plt.loglog(g.Rc, Exact3, c='k', ls=':') 
    plt.show()
    """


if __name__ == "__main__":
    test_reconstruct()
    test_adv_diff()