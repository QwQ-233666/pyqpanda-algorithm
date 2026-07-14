#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B92 量子密钥分发典型案例（基于两个非正交态 |0> 与 |+>）
==========================================================

演示内容：
  1. 无窃听时，B92 成功协商出一段安全密钥，并打印结论性轮次数与估计 QBER；
  2. 引入 Eve 的拦截-重发攻击后，QBER 显著升高，协议判定疑似窃听并中止。

运行方式（在 pyqpanda-algorithm/pyqpanda-algorithm 目录下）：
    python -m pyqpanda_alg.QKD.examples.demo_b92
或
    python pyqpanda_alg/QKD/examples/demo_b92.py
"""

import os
import sys

# 确保可以直接运行本脚本时也能找到 pyqpanda_alg 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pyqpanda_alg.QKD import B92


def main():
    print("=" * 64)
    print("B92 量子密钥分发演示（无窃听，基于非正交态 |0> 与 |+>）")
    print("=" * 64)
    b92 = B92(key_len=64, seed=42)
    key = b92.keygen(verbose=True)
    print("最终密钥长度 :", len(key))
    print("密钥(前 32 位):", key[:32], "...")
    print()

    print("=" * 64)
    print("B92 演示 Eve 拦截-重发攻击（eve_prob = 1.0）")
    print("=" * 64)
    eve = B92(key_len=64, eve_prob=1.0, seed=7)
    try:
        eve.keygen(verbose=True)
        print("错误：应当检测到窃听并中止，却没有！")
    except RuntimeError as e:
        print("协议已中止 :", e)


if __name__ == "__main__":
    main()
