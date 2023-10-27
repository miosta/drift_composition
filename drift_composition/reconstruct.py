# drift_composition: A 1D model for the advection and diffusion of molecules
# through protoplanetary discs.
#
# Copyright (C) 2023 R. Booth
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
Routines for doing polynomial reconstructions of data in the a finite-volume
framework.
"""
import numpy as np
import scipy.linalg

class BandedMatrix:
    """Wrapper for scipy banded matrix structure.

    Parameters
    ----------
    l_and_u : tuple(2)
        Specifies the number of lower and upper diagonal (l, u) elements in
        the banded structure.
    mat : array shape=(l+u+1, N)
        Elements of the banded matrix M, defined via:
            mat[u + i - j, j] == M[i,j]
    """
    def __init__(self, l_and_u, mat):
        self._lu = l_and_u
        self._mat = mat 

    @classmethod
    def create_from_weights(cls, stencil, w):
        """Create a scipy-banded matrix from the weights matrix"""
        l = - min(stencil)
        u = max(stencil)
        N = w.shape[0]
        banded = np.zeros((l+u+1,N), dtype=w.dtype)

        for i, s in enumerate(stencil):
            banded[u-s,max(s,0):min(N+s, N)] = w[max(-s,0):min(N-s,N),i]

        return cls((l, u), banded)

    def dot(self, x):
        """Compute the dot product between the banded matrix and the vector x.
        
        Note the unused elements of the banded matrix *must* be zero.
        """
        m = self._mat.reshape(self._mat.shape + (1,)*(len(x.shape)-1)) * x.reshape(-1, *x.shape)
        prod = np.zeros_like(x)
        N = self.size

        for s, v in zip(range(-self.l, self.u+1), m[::-1]):
            prod[max(-s,0):min(N-s,N)] += v[max(s,0):min(N+s,N)]
        
        return prod
    
    def __add__(self, other):
        """Add another banded matrix to this one
        
        Parameters
        ----------
        other : BandedMatrix
            Banded matrix to add to this one
        
        Returns
        -------
        total : BandedMatrix
            Result of adding other to self.
        """
        if self.size != other.size:
            raise ValueError("Both matrices must be the same shape")

        l = max(self.l, other.l)
        u = max(self.u, other.u)

        mat_sum = np.zeros((1+l+u, self.size), self._mat.dtype)

        for s in range(-self.l, self.u+1):
            mat_sum[u-s] = self._mat[self.u-s]

        for s in range(-other.l, other.u+1):
            mat_sum[u-s] += other._mat[other.u-s]

        return BandedMatrix((l,u), mat_sum)
    
    def __mul__(self, scale):
        """Multiplication by a scalar"""
        return BandedMatrix(self.l_and_u, self._mat * scale)
    
    def solve(self, rhs):
        """Solve the Linear system self * x = rhs for x
        
        Parameters
        ----------
        rhs : array, shape=(self.size) or shape=(self.size,M)
            RHS to solve the linear system for
        
        Returns
        -------
        x : array, shape=rhs.shape
            Solution to the linear system
        """
        return scipy.linalg.solve_banded(self.l_and_u, self._mat, rhs)
    
    def __sub__(self, other):
        return self + BandedMatrix(other.l_and_u, (-1)*other._mat)
    
    def __isub__(self, other):
        self = self - other
        return self
    
    def __iadd__(self, other):
        self = self + other
        return self
        
    def __imul__(self, scale):
        self = self * scale
        return self

    @property
    def l_and_u(self):
        return self._lu
    
    @property 
    def l(self):
        return self._lu[0]
    
    @property
    def u(self):
        return self._lu[1]
    
    @property
    def size(self):
        return self._mat.shape[1]


def _construct_volume_factors(xi, m, order, dtype):
    '''Evaluates the left-most matrix in Mignone's equation 21 (the matrix B
    in Appendix A)

    args:
        xi    : Cell edge locations
        m     : Radial scaling of the volumne element
        order : Order of the reconstruction
        dtype : numpy data-type to use
    '''
    beta = np.empty([len(xi) - 1, order], dtype=dtype)
    for n in range(order):
        beta[:, n] = np.diff(xi**(m+n+1)) / (m + n + 1.)

    beta.T[:] /= beta[:, 0].copy()

    return beta

def _construct_poly_derivs(xk, order, dtype):
    '''Evaluates the RHS of Mignone's equation 21, along with its derivatives

    args:
        xi    : Cell edge locations
        order : Order of the reconstruction
        dtype : numpy data-type to use        
    '''
    rhs = np.zeros([order, order], dtype=dtype)
    eta = np.power(xk, np.arange(order))
    rhs[:, 0] = eta
    for n in range(1, order):
        rhs[n:,n] = rhs[n-1:order-1,n-1]*range(n, order)
    return rhs

def _solve_FV_matrix_weights(xi, iL, iR, beta, max_deriv, dtype):
    '''Solves Mignone's equation 21, along with its derivatives'''
    order = 1 + iL + iR
    
    w = np.zeros([len(xi), max_deriv+1, order], dtype=dtype)
    for i in range(0, len(xi)):
        
        start  = max(0,       i-iL)
        end    = min(len(xi), i+iR+1)
        N      = end - start
        N_term = min(N, max_deriv+1)
        
        beta_i = beta[start:end,:N]
       
        # Solve for the coefficients
        rhs = _construct_poly_derivs(xi[i], N_term, dtype)
        w[i, :N, start-i+iL:end-i+iL] = np.linalg.solve(beta_i.T, rhs).T
        
    return w

def compute_centroids(xe, m):
    '''First order upwind reconstruction

    args:
        xe : Cell edge locations
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
    '''
    return ((m + 1) * np.diff(xe**(m+2))) / ((m + 2) * np.diff(xe**(m+1)))
    
def compute_FV_weights(xe, xj, m, stencil, max_deriv=None, dtype='f8'):
    '''Solves for the finite-volume interpolation weights.

    This code follows the methods outlined in Mignone (2014, JCoPh 270 784) to
    compute the weights needed to reconstruct a function and its derivatives
    to the specified locations. The polynomial is reconstructed to
    reproduce the averages of the cell and its neighbours.

    Note that the polynomial computed near the domain edges will be lower 
    order, which also reduces the number of derivatives available

    args:
        xe : locations of cell edges
        xj : Reconstruction points (one for each cell).
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
        stencil : Number of cells to the left and right of the target cell to 
                  use in the interpolation.

        max_deriv : maximum derivative level to calculate. If not specified,
                    return all meaningful derivatives.
    
        dtype : data type used for calculation, default = 'f8'

    returns:
        w : The weights used for reconstructing the function and its 1st
            iL+iR derivatives to the specified points.
            The shape is [len(xj), max_deriv+1, 1+iL+iR]
    '''
    # Order of the polynomial
    order = 1 + 2*stencil

    if max_deriv is None:
        max_deriv = order - 1
    elif max_deriv > order - 1:
        raise ValueError("Maximum derivative must be less than the order of the"
                         " polynomial fitted")
    
    # Setup the beta matrix of Mignone:
    beta =  _construct_volume_factors(xe, m, order, dtype)

    # Return the interpolated values
    return _solve_FV_matrix_weights(xj, stencil, stencil, beta, max_deriv, dtype)

def construct_FV_edge_weights(xi, m, stencil, max_deriv=None, dtype='f8'):
    '''Solves for the finite-volume interpolation weights.

    This code follows the methods outlined in Mignone (2014, JCoPh 270 784) to
    compute the weights needed to reconstruct a function and its derivatives
    to edges of computational cells. The polynomial is reconstructed to
    reproduce the averages of the cell and its neighbours.

    Note that the polynomial computed near the domain edges will be lower 
    order, which also reduces the number of derivatives available

    args:
        xi : locations of cell edges
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
        stencil : Number of cells to the left and right of the target cell to 
                  use in the interpolation.

        max_deriv : maximum derivative level to calculate. If not specified,
                    return all meaningful derivatives.
    
        dtype : data type used for calculation, default = 'f8'

    returns:
        wp, wm : The weights used for reconstructing the function and its 1st
                 iL+iR derivatives to the left and right of the cell edges. 
                 The shape is [len(xi)-1,max_deriv+1, 1+iL+iR]
    '''
    # Order of the polynomial
    order = 2*stencil
    iR = stencil
    iL = stencil-1

    if max_deriv is None:
        max_deriv = order - 1
    elif max_deriv > order - 1:
        raise ValueError("Maximum derivative must be less than the order of the"
                         " polynomial fitted")
    
    # Setup the beta matrix of Mignone:
    beta =  _construct_volume_factors(xi, m, order, dtype)

    # The matrix of extrapolations to the RHS of cell
    wp = _solve_FV_matrix_weights(xi[1:], iL, iR, beta, max_deriv, dtype)
    # The matrix of extrapolations to the LHS of cell
    wm = _solve_FV_matrix_weights(xi[:-1], iR, iL, beta, max_deriv, dtype)

    return wp, wm

def join_symmetric_stencil(wp, wm):
    '''Join together the weights in the case of a symmetric stencil.

    In this case both the wp and wm weights for the same edge are equal.
    '''
    if wp.shape != wm.shape:
        raise AttributeError("Error:Left/Right weights must have equal shapes")
    if wp.shape[1] % 2:
        raise AttributeError("Error: Weights must have an even stencil")

    w = np.concatenate([wm[:1], wp], axis=0)

    return w


def sparse_dot_product(w, x, stencil):
    """Compute the dot-product between the matrix specied by w and the
    vector x.
    """
    centroid = w.shape[0] == x.shape[0]
    return build_sparse_matrix(w, stencil, centroid).dot(x)


def build_sparse_matrix(w,stencil, centroid=False):
    '''Builds a sparse matrix from the weights for easy evaulation of the
    reconstruction.
    '''
    # Compute the shape of the stencil for centroid / edge values
    s = w.shape[1] // 2
    M = w.shape[0]
    if centroid:
        N = M
    else:
        N = M - 1

    # Seperate out the diagonals
    diags = []
    for j in range(w.shape[1]):
        ji = max(s-j, 0)
        je = min(N+s-j,M)
        diags.append(w[ji:je,j])

    # Create and return the sparse matrix
    return scipy.sparse.diags(diags, stencil, shape=(M,N))