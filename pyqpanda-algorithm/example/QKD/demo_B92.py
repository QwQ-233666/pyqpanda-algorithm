#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: B92 quantum key distribution.

B92 needs only two non-orthogonal states: Alice sends |0> for bit 0 and |+> for
bit 1.  Bob measures in a random basis and keeps only the *conclusive* outcomes
(about 25 % of rounds) that unambiguously identify Alice's bit.  As with BB84 an
eavesdropper is exposed by the resulting error rate.

Run:
    python demo_B92.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.QKD import B92
from pyqpanda_alg.QKD.QKD import QKDInsecureError


def report(title, qkd, n_raw):
    r = qkd.distribute(n_raw)
    print(f'\n[{title}]')
    print(f'  raw qubits         : {r.n_raw}')
    print(f'  conclusive (sifted): {len(r.alice_key)}  ({r.sift_rate:.1%})')
    print(f'  QBER               : {r.qber:.3f}   secure={r.secure}')
    print(f'  Alice sifted key   : {r.alice_key[:48]}...')
    print(f'  Bob   sifted key   : {r.bob_key[:48]}...')
    print(f'  keys identical     : {r.alice_key == r.bob_key}')


def main():
    print('=' * 64)
    print('B92 -- prepare-and-measure QKD (2 non-orthogonal states)')
    print('=' * 64)

    report('clean channel', B92(seed=42), 3000)
    report('eavesdropper on the line', B92(eavesdropper=True, seed=42), 3000)

    print('\n[keygen over a clean channel]')
    key = B92(seed=7).keygen(128)
    print(f'  128-bit shared key: {key}')

    print('\n[keygen with an eavesdropper -> aborts]')
    try:
        B92(eavesdropper=True, seed=7).keygen(128)
    except QKDInsecureError as e:
        print(f'  {e}')


if __name__ == '__main__':
    main()
