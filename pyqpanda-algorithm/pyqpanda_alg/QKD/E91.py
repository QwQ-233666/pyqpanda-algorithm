# ~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution#E91_protocol:_Artur_Ekert_.281991.29
"""
E91 (Ekert, 1991) -- entanglement-based QKD secured by a Bell inequality.

A source shares Bell pairs |Phi+> = (|00>+|11>)/sqrt2.  Alice measures her qubit
along one of three angles {0, pi/4, pi/2}; Bob along {pi/4, pi/2, 3pi/4}.

    * Key:      rounds where they happen to pick the *same physical angle*
                (Alice pi/4 = Bob pi/4, and Alice pi/2 = Bob pi/2) are perfectly
                correlated and become the shared key.
    * Security: the mismatched-angle rounds feed the CHSH quantity
                S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3).
                Genuine entanglement gives |S| = 2*sqrt2 ~ 2.83; any
                intercept-resend eavesdropper drives S back toward the classical
                bound of 2 (and simultaneously raises the QBER).
"""

import numpy as np

from .QKD import QKD

_PI = np.pi
A_ANGLES = (0.0, _PI / 4, _PI / 2)          # Alice's three settings
B_ANGLES = (_PI / 4, _PI / 2, 3 * _PI / 4)  # Bob's three settings
EVE_ANGLES = (0.0, _PI / 4, _PI / 2, 3 * _PI / 4)   # Eve guesses from the union

# rounds whose physical angles coincide -> raw key: (a_idx, b_idx)
_KEY_SETTINGS = {(1, 0), (2, 1)}
# CHSH uses Alice {0, pi/2} = idx {0, 2} and Bob {pi/4, 3pi/4} = idx {0, 2}
_CHSH_SETTINGS = ((0, 0), (0, 2), (2, 0), (2, 2))
_CHSH_SIGN = {(0, 0): +1, (0, 2): -1, (2, 0): +1, (2, 2): +1}


class E91(QKD):

    name = 'E91'
    sift_efficiency = 2.0 / 9.0     # 2 of the 9 equally-likely setting pairs are key rounds

    def _exchange(self, n_raw: int) -> dict:
        rng = self._rng
        a_idx = rng.integers(0, 3, n_raw)       # Alice's setting choices
        b_idx = rng.integers(0, 3, n_raw)       # Bob's setting choices
        e_ang = rng.choice(EVE_ANGLES, size=n_raw) if self.eavesdropper else None

        alice = np.empty(n_raw, dtype=int)
        bob = np.empty(n_raw, dtype=int)
        keep = np.zeros(n_raw, dtype=bool)

        # accumulate correlations for CHSH: sum of s_A*s_B and counts per setting
        corr_sum = {s: 0.0 for s in _CHSH_SETTINGS}
        corr_cnt = {s: 0 for s in _CHSH_SETTINGS}

        for i in range(n_raw):
            ai, bi = int(a_idx[i]), int(b_idx[i])
            if self.eavesdropper:
                a_out, eve_out = self._bell_measure(A_ANGLES[ai], float(e_ang[i]))
                b_out = self._resend(float(e_ang[i]), eve_out, B_ANGLES[bi])
            else:
                a_out, b_out = self._bell_measure(A_ANGLES[ai], B_ANGLES[bi])
            b_out = self._maybe_flip(b_out)

            alice[i], bob[i] = a_out, b_out
            if (ai, bi) in _KEY_SETTINGS:
                keep[i] = True
            if (ai, bi) in corr_sum:
                corr_sum[(ai, bi)] += (1 - 2 * a_out) * (1 - 2 * b_out)
                corr_cnt[(ai, bi)] += 1

        S = 0.0
        for s in _CHSH_SETTINGS:
            E = corr_sum[s] / corr_cnt[s] if corr_cnt[s] else 0.0
            S += _CHSH_SIGN[s] * E

        return dict(alice=alice, bob=bob, keep=keep,
                    extra={'chsh_S': float(S)})
