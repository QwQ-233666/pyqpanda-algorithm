#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E91 量子密钥分发典型案例
==========================

演示内容：
  1. 无窃听时，E91 借助最大纠缠 Bell 对协商出密钥，并打印 CHSH 的 S 值
     （理想情况下 |S| ≈ 2√2 ≈ 2.828 > 2，验证量子关联）；
  2. 引入 Eve 对 Bob 一路粒子的拦截-重发攻击后，纠缠被破坏，|S| 跌到经典
     上界 2 附近、QBER 升高，协议判定不安全并中止。

运行方式（在 pyqpanda-algorithm/pyqpanda-algorithm 目录下）：
    python -m pyqpanda_alg.QKD.examples.demo_e91
或
    python pyqpanda_alg/QKD/examples/demo_e91.py
"""

import os
import sys

# 确保可以直接运行本脚本时也能找到 pyqpanda_alg 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pyqpanda_alg.QKD import E91


def main():
    print("=" * 64)
    print("E91 量子密钥分发演示（无窃听，基于纠缠 + CHSH 校验）")
    print("=" * 64)
    e91 = E91(key_len=64, seed=42)
    key = e91.keygen(verbose=True)
    print("最终密钥长度 :", len(key))
    print("密钥(前 32 位):", key[:32], "...")
    print()

    print("=" * 64)
    print("E91 演示 Eve 拦截-重发攻击（eve_prob = 1.0）")
    print("=" * 64)
    eve = E91(key_len=64, eve_prob=1.0, seed=7)
    try:
        eve.keygen(verbose=True)
        print("错误：应当检测到窃听并中止，却没有！")
    except RuntimeError as e:
        print("协议已中止 :", e)


if __name__ == "__main__":
    main()
