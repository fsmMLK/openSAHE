#!/bin/python
# -*- coding: utf-8 -*-
"""
definition of basic sparse functions
"""

from __future__ import division, print_function

import numpy as np
from scipy import sparse as scipySparse

from PyPardisoProject.pypardiso.pardiso_wrapper import PyPardisoSolver


class myPardisoSparse():
    def __init__(self, matrix, mtype=11, nCoresPardiso=2):
        """

        Parameters
        ----------
        matrix: numpy array
            this matrix can be a scipy sparse matrix or dense matrix.
        """

        if scipySparse.issparse(matrix):
            if scipySparse.isspmatrix_csr(matrix):
                self.mat = matrix
            else:
                self.mat = matrix.tocsr()
        else:
            self.mat = scipySparse.csr_matrix(matrix)

        self.solver = PyPardisoSolver(mtype=mtype, phase=13, size_limit_storage=5e7)

        self.solver._check_A(self.mat)

        if not self.solver._is_already_factorized(self.mat):
            self.solver.factorize(self.mat)

        # set number of threads
        # max_threads = pyMKL.mkl_get_max_threads()
        # self.nThreadsPardiso = min(max_threads, nCoresPardiso)
        # #pyMKL.mkl_set_num_threads(self.nThreadsPardiso)
        # from ctypes import Structure, POINTER, c_int, c_char_p
        # pyMKL.MKLlib.mkl_set_num_threads(c_int(self.nThreadsPardiso))
        # pyMKL.MKLlib.domain_set_num_threads(self.nThreadsPardiso, domain='pardiso')
        # print("MKL: max. Threads: {}".format(max_threads) + ". Using %d Threads..." % self.nThreadsPardiso )

        self.nCoresPardiso = nCoresPardiso
        self.solver.set_num_threads(nCoresPardiso)
        print("    -> MKL: Using %d Threads..." % self.solver.get_max_threads())

    def invert(self):
        b = np.eye(self.mat.shape[0])
        x = self.solve(b)
        self.solver.free_memory(everything=True)
        return x

    def solve(self, b):
        """
        This function mimics scipy.sparse.linalg.spsolve, but uses the Pardiso solver instead of SuperLU/UMFPACK

            solve Ax=b for x

            --- Parameters ---
            b: numpy ndarray
               right-hand side(s), b.shape[0] needs to be the same as A.shape[0]

            --- Returns ---
            x: numpy ndarray
               solution of the system of linear equations, same shape as b (but returns shape (n,) if b has shape (n,1))

        """
        # scipy spsolve always returns vectors with shape (n,) indstead of (n,1)
        return self.solver.solve(self.mat, b).squeeze()
