# ~ref: https://en.wikipedia.org/wiki/SARG04
"""
SARG04 (Scarani, Acin, Ribordy & Gisin, 2004) -- same four states as BB84 but a
different sifting rule that hardens it against photon-number-splitting attacks.

Alice sends one of the four BB84 states; the *key bit is the basis she used*
(0 = Z, 1 = X), not the state value.  During sifting she does not reveal the
basis; instead she announces an unordered pair made of the state she sent and one
decoy from the conjugate basis.  Bob, who measured in a random basis, can identify
the sent state only when his outcome is orthogonal to -- and therefore excludes --
the decoy:

    conclusive  <=>  Bob measured in the conjugate basis AND his outcome is the
                     opposite value of the decoy.

His own outcome can never be orthogonal to the true state, so the exclusion is
unambiguous and he recovers Alice's basis (the key bit).  The ideal sift rate is
~25 %.
"""

import numpy as np

from .QKD import QKD


class SARG04(QKD):

    name = 'SARG04'
    sift_efficiency = 0.25

    def _exchange(self, n_raw: int) -> dict:
        rng = self._rng
        send_basis = rng.integers(0, 2, n_raw)   # Alice's basis == her KEY BIT
        send_val = rng.integers(0, 2, n_raw)      # which state within that basis
        decoy_val = rng.integers(0, 2, n_raw)     # decoy state (from the conjugate basis)
        b_basis = rng.integers(0, 2, n_raw)       # Bob's random measurement basis
        eve = self._eve_mask(n_raw)
        e_basis = rng.integers(0, 2, n_raw) if np.any(eve) else None

        alice = send_basis.copy()                 # the key bit Alice intends
        bob = np.zeros(n_raw, dtype=int)
        keep = np.zeros(n_raw, dtype=bool)
        for i in range(n_raw):
            prep_basis, prep_bit = int(send_basis[i]), int(send_val[i])
            if eve[i]:
                eve_bit = self._pm(prep_basis, prep_bit, int(e_basis[i]))
                prep_basis, prep_bit = int(e_basis[i]), eve_bit

            out = self._maybe_flip(self._pm(prep_basis, prep_bit, int(b_basis[i])))

            # Bob's outcome (b_basis, out) excludes an announced state when it is
            # orthogonal to it: same basis, opposite value. He concludes the
            # *other*, non-excluded state and reads off its basis as the key bit.
            other_basis = 1 - int(send_basis[i])
            excl_sent = (int(b_basis[i]) == int(send_basis[i])) and (out != int(send_val[i]))
            excl_decoy = (int(b_basis[i]) == other_basis) and (out != int(decoy_val[i]))
            if excl_decoy and not excl_sent:
                bob[i], keep[i] = int(send_basis[i]), True      # concluded Alice's state
            elif excl_sent and not excl_decoy:
                bob[i], keep[i] = other_basis, True             # concluded the decoy (an error)

        return dict(alice=alice, bob=bob, keep=keep,
                    extra={'eve_intercepts': int(eve.sum()),
                           'eve_rate': float(eve.mean())})
