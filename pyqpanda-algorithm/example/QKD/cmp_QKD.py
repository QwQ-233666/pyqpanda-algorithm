#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the five QKD protocols side by side.

Runs BB84, B92, E91, BBM92 and SARG04 over a clean channel and again with an
intercept-resend eavesdropper, then reports sift efficiency and the quantum
bit-error rate (QBER) for each.  In every case the QBER stays at 0 without an
eavesdropper and rises well above the ~11 % security threshold once Eve is on the
line -- the signature that lets the parties detect her.

Outputs (written to ./output):
    cmp_QKD.png  -- sift-efficiency and QBER (clean vs eavesdropped) bar charts

Run:
    python cmp_QKD.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.QKD import BB84, B92, E91, BBM92, SARG04

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'output'

PROTOCOLS = [BB84, B92, E91, BBM92, SARG04]
N_RAW = 6000
THRESHOLD = 0.11
EVE_PROB = 1.0


def main():
    names, sift, qber_clean, qber_eve = [], [], [], []
    print(f'{"protocol":8s} {"sift":>7s} {"QBER(clean)":>12s} {"QBER(Eve)":>10s}  secure(clean/Eve)')
    print('-' * 60)
    for P in PROTOCOLS:
        clean = P(seed=1).distribute(N_RAW)
        eve = P(eve_prob=EVE_PROB, seed=1).distribute(N_RAW)
        names.append(P.__name__)
        sift.append(clean.sift_rate)
        qber_clean.append(clean.qber)
        qber_eve.append(eve.qber)
        chsh = ''
        if 'chsh_S' in clean.extra:
            chsh = f"   CHSH S: clean={clean.extra['chsh_S']:+.2f} Eve={eve.extra['chsh_S']:+.2f}"
        print(f'{P.__name__:8s} {clean.sift_rate:6.1%} {clean.qber:12.3f} {eve.qber:10.3f}'
              f'   {clean.secure}/{eve.secure}{chsh}')

    OUT_DIR.mkdir(exist_ok=True)
    x = range(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(x, sift, color='#4c72b0')
    ax1.set_xticks(list(x)); ax1.set_xticklabels(names, rotation=20)
    ax1.set_ylabel('sift efficiency')
    ax1.set_title('Sift efficiency (kept fraction)')
    ax1.set_ylim(0, 0.6)
    for i, v in zip(x, sift):
        ax1.text(i, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=9)

    w = 0.38
    ax2.bar([i - w / 2 for i in x], qber_clean, width=w, label='clean channel', color='#55a868')
    ax2.bar([i + w / 2 for i in x], qber_eve, width=w,
            label=f'Eve intercepts {EVE_PROB:.0%}', color='#c44e52')
    ax2.axhline(THRESHOLD, ls='--', color='k', lw=1, label=f'security threshold {THRESHOLD:.0%}')
    ax2.set_xticks(list(x)); ax2.set_xticklabels(names, rotation=20)
    ax2.set_ylabel('QBER')
    ax2.set_title('QBER: clean vs eavesdropped')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / 'cmp_QKD.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'\nSaved comparison figure to {path}')


if __name__ == '__main__':
    main()
