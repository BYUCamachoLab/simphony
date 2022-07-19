# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.connect
================

Code for s-parameter matrix cascading uses the scikit-rf implementation. Per
their software license, the copyright notice is reproduced below:


Copyright (c) 2010, Alexander Arsenovic
All rights reserved.

Copyright (c) 2017, scikit-rf Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the scikit-rf nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

from simphony.tools import add_polar, mul_polar

# Functions operating on s-parameter matrices
def connect_s(A, k, B, l):
    """
    connect two n-port networks' s-matrices together.
    specifically, connect port `k` on network `A` to port `l` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2). This
    function operates on, and returns s-matrices. The function
    :func:`connect` operates on :class:`Network` types.
    Parameters
    -----------
    A : :class:`numpy.ndarray`
            S-parameter matrix of `A`, shape is fxnxn(x2)
    k : int
            port index on `A` (port indices start from 0)
    B : :class:`numpy.ndarray`
            S-parameter matrix of `B`, shape is fxnxn(x2)
    l : int
            port index on `B`
    Returns
    -------
    C : :class:`numpy.ndarray`
        new S-parameter matrix
    Notes
    -------
    internally, this function creates a larger composite network
    and calls the  :func:`innerconnect_s` function. see that function for more
    details about the implementation
    See Also
    --------
        connect : operates on :class:`Network` types
        innerconnect_s : function which implements the connection
            connection algorithm
    """

    if k > A.shape[-1] - 1 or l > B.shape[-1] - 1:
        raise (ValueError("port indices are out of range"))

    C = create_block_diagonal(A, B)
    nA = A.shape[1]  # num ports on A

    # call innerconnect_s() on composit matrix C
    return innerconnect_s(C, k, nA + l)


def create_block_diagonal(A, B):
    """merges an fxnxn(x2) matrix with an fxmxm(x2) matrix to form a fx(n+m)x(n+m)(x2)
    block diagonal matrix."""
    nf = A.shape[0]  # num frequency points
    nA = A.shape[1]  # num ports on A
    nB = B.shape[1]  # num ports on B
    nC = nA + nB  # num ports on C

    # if complex values are in rectangular, convert to polar
    if A.ndim == 3:
        A = np.stack((np.abs(A), np.angle(A)), axis=-1)

    if B.ndim == 3:
        B = np.stack((np.abs(B), np.angle(B)), axis=-1)

    # create composite matrix, appending each sub-matrix diagonally
    C = np.zeros((nf, nC, nC, 2))
    C[:, :nA, :nA] = A.copy()
    C[:, nA:, nA:] = B.copy()

    return C


def innerconnect_s(A, k, l):
    """
    connect two ports of a single n-port network's s-matrix.
    Specifically, connect port `k`  to port `l` on `A`. This results in
    a (n-2)-port network.  This     function operates on, and returns
    s-matrices. The function :func:`innerconnect` operates on
    :class:`Network` types.
    Parameters
    -----------
    A : :class:`numpy.ndarray`
        S-parameter matrix of `A`, shape is fxnxnx2
    k : int
        port index on `A` (port indices start from 0)
    l : int
        port index on `A`
    Returns
    -------
    C : :class:`numpy.ndarray`
            new S-parameter matrix
    Notes
    -----
    The algorithm used to calculate the resultant network is called a
    'sub-network growth',  can be found in [#]_. The original paper
    describing the  algorithm is given in [#]_.
    References
    ----------
    .. [#] Compton, R.C.; , "Perspectives in microwave circuit analysis," Circuits and Systems, 1989., Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16 Aug 1989. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167
    .. [#] Filipsson, Gunnar; , "A New General Computer Algorithm for S-Matrix Calculation of Interconnected Multiports," Microwave Conference, 1981. 11th European , vol., no., pp.700-704, 7-11 Sept. 1981. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4131699&isnumber=4131585
    """

    if k > A.shape[1] - 1 or l > A.shape[1] - 1:
        raise (ValueError("port indices are out of range"))

    nA = A.shape[1]  # num of ports on input s-matrix
    # create an empty s-matrix, to store the result
    C = np.zeros(A.shape)

    # loop through ports and calulates resultant s-parameters
    for h in range(A.shape[0]):
        for i in range(nA):
            for j in range(nA):
                term1 = mul_polar(
                    mul_polar(A[h, k, j], A[h, i, l]),
                    (1 - A[h, l, k, 0], A[h, l, k, 1]),
                )
                term2 = mul_polar(
                    mul_polar(A[h, l, j], A[h, i, k]),
                    (1 - A[h, k, l, 0], A[h, k, l, 1]),
                )
                term3 = mul_polar(mul_polar(A[h, k, j], A[h, l, l]), A[h, i, k])
                term4 = mul_polar(mul_polar(A[h, l, j], A[h, k, k]), A[h, i, l])
                term5 = mul_polar(
                    (1 - A[h, k, l, 0], A[h, k, l, 1]),
                    (1 - A[h, l, k, 0], A[h, l, k, 1]),
                )
                term6 = mul_polar(A[h, k, k], A[h, l, l])
                term7 = add_polar(add_polar(add_polar(term1, term2), term3), term4)
                term8 = add_polar(term5, (-term6[0], term6[1]))
                term9 = (term7[0] / term8[0], term7[1] - term8[1])
                C[h, i, j] = add_polar(A[h, i, j], term9)

    # remove ports that were `connected`
    C = np.delete(C, (k, l), 1)
    C = np.delete(C, (k, l), 2)

    return C
