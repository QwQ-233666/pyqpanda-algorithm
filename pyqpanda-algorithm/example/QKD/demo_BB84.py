#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB84 量子密钥分发典型案例
============================

演示内容：
  1. 无窃听时，BB84 成功协商出一段安全密钥，并打印估计的 QBER；
  2. 引入 Eve 的拦截-重发攻击后，QBER 显著升高，协议判定疑似窃听并中止。

运行方式（在 pyqpanda-algorithm/pyqpanda-algorithm 目录下）：
    python -m pyqpanda_alg.QKD.examples.demo_bb84
或
    python pyqpanda_alg/QKD/examples/demo_bb84.py
"""

import os
import sys

# 确保可以直接运行本脚本时也能找到 pyqpanda_alg 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pyqpanda_alg.QKD import BB84


def main():
    print("=" * 64)
    print("BB84 量子密钥分发演示（无窃听）")
    print("=" * 64)
    bb84 = BB84(key_len=64, seed=42)
    key = bb84.keygen(verbose=True)
    print("最终密钥长度 :", len(key))
    print("密钥(前 32 位):", key[:32], "...")
    print()

    print("=" * 64)
    print("BB84 演示 Eve 拦截-重发攻击（eve_prob = 1.0）")
    print("=" * 64)
    eve = BB84(key_len=64, eve_prob=1.0, seed=7)
    try:
        eve.keygen(verbose=True)
        print("错误：应当检测到窃听并中止，却没有！")
    except RuntimeError as e:
        print("协议已中止 :", e)


if __name__ == "__main__":
    main()
