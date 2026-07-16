# ~ref: https://en.wikipedia.org/wiki/BBM92_protocol
"""
BBM92 (Bennett, Brassard & Mermin, 1992) -- the entanglement-based twin of BB84.

Instead of Alice preparing states, a shared source emits Bell pairs
|Phi+> = (|00>+|11>)/sqrt2.  Alice and Bob each measure their qubit in a random
basis (Z or X).  Since |Phi+> is perfectly correlated in *both* bases
( = (|++>+|-->)/sqrt2 ), whenever their bases agree their outcomes are identical,
yielding the raw key.  An eavesdropper who intercepts and resends one arm breaks
the entanglement and shows up as errors, exactly as in BB84.
"""

import numpy as np

from .QKD import QKD, Z_BASIS, X_BASIS, _HALF_PI


class BBM92(QKD):

    name = 'BBM92'
    sift_efficiency = 0.5      # bases agree half of the time

    @staticmethod
    def _rot(basis: int) -> float:
        return _HALF_PI if basis == X_BASIS else 0.0     # Z -> 0, X -> pi/2

    def _exchange(self, n_raw: int) -> dict:
        rng = self._rng
        a_basis = rng.integers(0, 2, n_raw)     # Alice's random bases
        b_basis = rng.integers(0, 2, n_raw)     # Bob's random bases
        eve = self._eve_mask(n_raw)
        e_basis = rng.integers(0, 2, n_raw) if np.any(eve) else None

        alice = np.empty(n_raw, dtype=int)
        bob = np.empty(n_raw, dtype=int)
        for i in range(n_raw):
            a_rot, b_rot = self._rot(int(a_basis[i])), self._rot(int(b_basis[i]))
            if eve[i]:
                # Eve measures Bob's arm, collapsing the pair, then resends
                e_rot = self._rot(int(e_basis[i]))
                a_out, eve_out = self._bell_measure(a_rot, e_rot)
                b_out = self._resend(e_rot, eve_out, b_rot)
            else:
                a_out, b_out = self._bell_measure(a_rot, b_rot)
            alice[i] = a_out
            bob[i] = self._maybe_flip(b_out)

        keep = a_basis == b_basis                # sift: matching bases
        return dict(alice=alice, bob=bob, keep=keep,
                    extra={'eve_intercepts': int(eve.sum()),
                           'eve_rate': float(eve.mean())})
