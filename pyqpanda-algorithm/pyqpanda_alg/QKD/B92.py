# ~ref: https://en.wikipedia.org/wiki/B92_protocol
"""
B92 (Bennett, 1992) -- a minimalist prepare-and-measure protocol that needs only
two non-orthogonal states.

Alice encodes bit 0 as |0> and bit 1 as |+>.  Bob measures each qubit in a random
basis (Z or X) and keeps only *conclusive* outcomes:

    * measured in Z and got |1>  =>  the state cannot have been |0>, so Alice sent
      |+>  =>  key bit 1;
    * measured in X and got |->  =>  the state cannot have been |+>, so Alice sent
      |0>  =>  key bit 0.

All other outcomes are inconclusive and discarded, giving a ~25 % sift rate in the
ideal case.  Because the two states are non-orthogonal, no measurement can
distinguish them deterministically -- which is exactly what keeps Eve out.
"""

import numpy as np

from .QKD import QKD, Z_BASIS, X_BASIS


class B92(QKD):

    name = 'B92'
    sift_efficiency = 0.25     # conclusive a quarter of the time

    def _exchange(self, n_raw: int) -> dict:
        rng = self._rng
        a_bits = rng.integers(0, 2, n_raw)      # Alice's key bits
        b_basis = rng.integers(0, 2, n_raw)     # Bob's random measurement bases
        e_basis = rng.integers(0, 2, n_raw) if self.eavesdropper else None

        bob = np.zeros(n_raw, dtype=int)
        keep = np.zeros(n_raw, dtype=bool)
        for i in range(n_raw):
            # bit 0 -> |0> (Z, value 0); bit 1 -> |+> (X, value 0)
            prep_basis = X_BASIS if a_bits[i] == 1 else Z_BASIS
            prep_bit = 0
            if self.eavesdropper:
                eve_bit = self._pm(prep_basis, prep_bit, int(e_basis[i]))
                prep_basis, prep_bit = int(e_basis[i]), eve_bit

            out = self._maybe_flip(self._pm(prep_basis, prep_bit, int(b_basis[i])))

            if b_basis[i] == Z_BASIS and out == 1:
                bob[i], keep[i] = 1, True       # |1> in Z rules out |0>  => bit 1
            elif b_basis[i] == X_BASIS and out == 1:
                bob[i], keep[i] = 0, True       # |-> in X rules out |+>  => bit 0
            # otherwise inconclusive -> discarded

        return dict(alice=a_bits, bob=bob, keep=keep)
