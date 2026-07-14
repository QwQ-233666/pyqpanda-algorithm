#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBM92 量子密钥分发典型案例（基于 EPR 纠缠对 |Φ+>，BB84 的纠缠版本）
====================================================================

演示内容：
  1. 无窃听时，BBM92 利用纠缠对的同基相关性成功协商出密钥，并打印 QBER；
  2. 引入 Eve 对 Bob 一路粒子的拦截-重发攻击后，同基相关性被破坏、QBER 升高，
     协议判定疑似窃听并中止。

运行方式（在 pyqpanda-algorithm/pyqpanda-algorithm 目录下）：
    python -m pyqpanda_alg.QKD.examples.demo_bbm92
或
    python pyqpanda_alg/QKD/examples/demo_bbm92.py
"""

import os
import sys

# 确保可以直接运行本脚本时也能找到 pyqpanda_alg 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pyqpanda_alg.QKD import BBM92


def main():
    print("=" * 64)
    print("BBM92 量子密钥分发演示（无窃听，基于 EPR 纠缠对 |Φ+>）")
    print("=" * 64)
    bbm92 = BBM92(key_len=64, seed=42)
    key = bbm92.keygen(verbose=True)
    print("最终密钥长度 :", len(key))
    print("密钥(前 32 位):", key[:32], "...")
    print()

    print("=" * 64)
    print("BBM92 演示 Eve 拦截-重发攻击（eve_prob = 1.0）")
    print("=" * 64)
    eve = BBM92(key_len=64, eve_prob=1.0, seed=7)
    try:
        eve.keygen(verbose=True)
        print("错误：应当检测到窃听并中止，却没有！")
    except RuntimeError as e:
        print("协议已中止 :", e)


if __name__ == "__main__":
    main()
