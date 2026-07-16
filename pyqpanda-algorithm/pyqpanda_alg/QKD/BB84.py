# ~ref: https://en.wikipedia.org/wiki/BB84
"""
BB84 (Bennett & Brassard, 1984) -- the original prepare-and-measure protocol.

For every qubit Alice draws a random bit and a random basis (Z: |0>,|1> or
X: |+>,|->), prepares the corresponding state and sends it.  Bob measures in his
own random basis.  During sifting they keep only the rounds where their bases
match; on those the two bits are identical in an ideal channel.  An
intercept-resend eavesdropper, forced to guess the basis, corrupts 25 % of the
sifted bits and is exposed by the QBER.
"""

import numpy as np

from .QKD import QKD, Z_BASIS, X_BASIS


class BB84(QKD):

    name = 'BB84'
    sift_efficiency = 0.5      # bases agree half of the time

    def _exchange(self, n_raw: int) -> dict:
        rng = self._rng
        a_bits = rng.integers(0, 2, n_raw)      # Alice's random bits
        a_basis = rng.integers(0, 2, n_raw)     # Alice's random bases
        b_basis = rng.integers(0, 2, n_raw)     # Bob's random bases
        eve = self._eve_mask(n_raw)
        e_basis = rng.integers(0, 2, n_raw) if np.any(eve) else None

        bob = np.empty(n_raw, dtype=int)
        for i in range(n_raw):
            prep_basis, prep_bit = int(a_basis[i]), int(a_bits[i])
            if eve[i]:
                # Eve measures in a random basis and resends what she found
                eve_bit = self._pm(prep_basis, prep_bit, int(e_basis[i]))
                prep_basis, prep_bit = int(e_basis[i]), eve_bit
            out = self._pm(prep_basis, prep_bit, int(b_basis[i]))
            bob[i] = self._maybe_flip(out)

        keep = a_basis == b_basis                # sift: matching bases
        return dict(alice=a_bits, bob=bob, keep=keep,
                    extra={'eve_intercepts': int(eve.sum()),
                           'eve_rate': float(eve.mean())})
