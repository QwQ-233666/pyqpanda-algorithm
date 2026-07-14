# ~ref: https://en.wikipedia.org/wiki/BBM92_protocol
# ~ref: C. H. Bennett, G. Brassard, N. D. Mermin, "Quantum cryptography without
#       Bell's theorem", Phys. Rev. Lett. 68, 557 (1992).
"""
BBM92 量子密钥分发协议（Bennett-Brassard-Mermin, 1992）。

BBM92 是 **BB84 的纠缠版本**：用 EPR 纠缠对取代「Alice 制备、Bob 测量」的
单光子源。它与 E91 同源（都基于最大纠缠 Bell 态），但安全性检验方式不同：

  * 一个可信（或半可信）源为每一轮分发一个最大纠缠 Bell 对
    |Φ+> = (|00> + |11>)/√2，分别发给 Alice 与 Bob；
  * 双方各自**独立随机**选择测量基（Z 基 {|0>,|1>} 或 X 基 {|+>,|->}）；
  * **密钥提取**：当双方选了「相同基」时，对 |Φ+> 的测量结果完全相关
    （Bob 的比特与 Alice 一致），用 Alice 的比特作为原始密钥；
  * **安全性检验**：与 BB84 类似——从「同基」的已筛选密钥中**公开牺牲**一小
    部分比特来估计 QBER。这与 E91 用 CHSH 不等式检验纠缠不同：
    BBM92 不依赖 Bell 不等式，而是直接做误码率估计。

与 BB84 的主要差异：BB84 是「制备-测量」型（Prepare & Measure），BBM92 是
「纠缠分发」型（Entanglement-based）。与 E91 的主要差异：E91 用 CHSH 违背来
同时完成密钥提取与安全性证明；BBM92 则更贴近 BB84，用同基相关 + QBER 估计。

本实现用 pyqpanda3 的 CPU 模拟器真实搭建 Bell 态与（含 Eve 拦截-重发）测量
线路，通过后处理（sifting / QBER / 隐私放大）得到最终密钥。
"""

from typing import Optional

import numpy as np
from .QKD import QKD
from pyqpanda3.core import QCircuit, H, CNOT, X


class BBM92(QKD):
    """
    BBM92 协议实现（基于 EPR 纠缠对 |Φ+>）。

    Parameters
        key_len : int
            期望最终密钥长度（比特）。
        qber_threshold : float
            QBER 安全阈值，默认 0.11（与 BB84 一致的安全上限）。
        eve_prob : float
            Eve 对 Bob 那一路粒子做拦截-重发攻击的概率（0~1）。非 0 时会破坏
            同基相关性、抬升 QBER，使协议中止。
        seed : int or None
            随机种子，便于复现。
    """

    def __init__(self, key_len: int = 128, qber_threshold: float = 0.11,
                 eve_prob: float = 0.0, seed: int = None):
        super().__init__(key_len=key_len, qber_threshold=qber_threshold, seed=seed)
        if not (0.0 <= eve_prob <= 1.0):
            raise ValueError("eve_prob 应位于 [0, 1]")
        self.eve_prob = float(eve_prob)

    # ----------------------------- 量子线路工具 ----------------------------- #
    def _bell(self) -> QCircuit:
        """制备 Bell 态 |Φ+> = (|00> + |11>)/√2。"""
        qc = QCircuit()
        qc << H(0) << CNOT(0, 1)
        return qc

    def _measure_bell(self, a_basis: int, b_basis: int) -> tuple:
        """对 Bell 源的两个粒子分别在 a_basis、b_basis 下测量，返回 (a_bit, b_bit)。"""
        qc = self._bell()
        if a_basis == QKD.BASIS_X:
            qc << H(0)
        if b_basis == QKD.BASIS_X:
            qc << H(1)
        out = self._sample_outcome(qc, 2)
        return int(out[0]), int(out[1])

    def _measure_bell_eve(self, a_basis: int, b_basis: int, e_basis: int) -> tuple:
        """
        Eve 拦截 Bob 那一路：先测自己那半得到 e_bit，再用 (e_bit, e_basis)
        重新制备一个粒子发给 Bob，Bob 再按 b_basis 测量。返回 (a_bit, b_bit)。
        """
        # 1) Alice 测自己那半（a_basis），Eve 测原 Bob 那半（e_basis）
        qc1 = self._bell()
        if a_basis == QKD.BASIS_X:
            qc1 << H(0)
        if e_basis == QKD.BASIS_X:
            qc1 << H(1)
        out1 = self._sample_outcome(qc1, 2)
        a_bit = int(out1[0])
        e_bit = int(out1[1])
        # 2) Eve 重新制备 (e_bit, e_basis) 交给 Bob，Bob 按 b_basis 测量
        qc2 = QCircuit()
        if e_bit:
            qc2 << X(0)
        if e_basis == QKD.BASIS_X:
            qc2 << H(0)
        if b_basis == QKD.BASIS_X:
            qc2 << H(0)
        b_bit = int(self._sample_outcome(qc2, 1))
        return a_bit, b_bit

    # ------------------------------- 密钥生成 ------------------------------- #
    def keygen(self, nlen: Optional[int] = None, verbose: bool = False) -> str:
        """
        运行 BBM92 协议并返回最终密钥（二进制字符串）。

        参数
            nlen   : int, optional
                最终密钥长度；省略时使用构造时的 key_len。
            verbose: bool
                为 True 时打印同基轮次数、QBER 等中间信息。
        返回
            key_str: str
                长度为 nlen 的二进制字符串。
        异常
            RuntimeError: 当估计的 QBER 超过安全阈值（疑似窃听）时抛出。
        """
        if nlen is None:
            nlen = self.key_len
        # 同基轮次约占 1/2，留足余量做 QBER 测试与隐私放大
        n_send = max(int(round(nlen * 8)), 256)

        alice_bits = []
        bob_bits = []
        alice_bases = []
        bob_bases = []

        for _ in range(n_send):
            a_basis = int(self.rng.integers(0, 2))
            b_basis = int(self.rng.integers(0, 2))

            if self.eve_prob > 0 and self.rng.random() < self.eve_prob:
                e_basis = int(self.rng.integers(0, 2))
                a_bit, b_bit = self._measure_bell_eve(a_basis, b_basis, e_basis)
            else:
                a_bit, b_bit = self._measure_bell(a_basis, b_basis)

            # sifting：只保留双方选了相同基的轮次（对 |Φ+> 结果完全相关）
            if a_basis == b_basis:
                alice_bits.append(a_bit)
                bob_bits.append(b_bit)
                alice_bases.append(a_basis)
                bob_bases.append(b_basis)

        sift_a = np.asarray(alice_bits, dtype=int)
        sift_b = np.asarray(bob_bits, dtype=int)
        if len(sift_a) == 0:
            raise RuntimeError("sifting 后没有保留任何比特，请增大 n_send 或 key_len")

        # QBER 估计：理想情况下同基轮次双方完全一致（QBER=0）
        qber, keep = self._estimate_qber(sift_a, sift_b)
        if verbose:
            print("[BBM92] 同基轮次=%d, 估计 QBER=%.3f, 阈值=%.3f"
                  % (len(sift_a), qber, self.qber_threshold))
        if not keep:
            raise RuntimeError(
                "检测到疑似窃听：QBER=%.3f 超过安全阈值 %.3f，协议中止"
                % (qber, self.qber_threshold)
            )

        # 纠错近似：去掉 QBER 测试抽样位，剩余比特做隐私放大
        n = len(sift_a)
        test_k = max(1, int(round(n * 0.2)))
        test_k = min(test_k, n)
        test_idx = set(self.rng.choice(n, size=test_k, replace=False).tolist())
        remaining = np.array([b for i, b in enumerate(sift_a) if i not in test_idx],
                             dtype=int)
        if len(remaining) < nlen:
            raise RuntimeError(
                "可用于隐私放大的比特不足（%d < %d），请增大 key_len"
                % (len(remaining), nlen)
            )

        final = self._privacy_amplification(remaining, final_len=nlen)
        return self._bits_to_str(final)
