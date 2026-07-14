# ~ref: https://en.wikipedia.org/wiki/BB84
"""
BB84 量子密钥分发协议（Bennett & Brassard, 1984）。

BB84 是第一个（也是最广为人知的）QKD 协议：
  * Alice 为每一位随机选取一个比特值，并随机在 Z 基 {|0>,|1>} 或
    X 基 {|+>,|->} 中制备一个单光子态发送给 Bob；
  * Bob 对收到的光子随机选一个基测量，得到自己的比特；
  * 双方公开比对各自使用的基，只保留「基相同」的轮次（sifting）；
  * 抽样估计 QBER，超阈值则中止；
  * 最后做隐私放大得到最终密钥。

本实现用 pyqpanda3 的 CPU 模拟器真正搭建「制备 -> 测量」单比特线路，
通过线路的概率分布采样得到测量比特；经典后处理（sifting / QBER /
隐私放大）复用 `QKD` 基类。同时支持插入 Eve 的拦截-重发攻击以演示窃听
检测。
"""

from typing import Optional

from .QKD import QKD
from pyqpanda3.core import QCircuit, X, H
import numpy as np


class BB84(QKD):
    """
    BB84 协议实现。

    Parameters
        key_len : int
            期望最终密钥长度（比特）。
        qber_threshold : float
            QBER 安全阈值，默认 0.11（约 11%，BB84 安全上限）。
        eve_prob : float
            Eve 进行拦截-重发攻击的概率（0~1）。0 表示无窃听。
            非 0 时由于拦截-重发引入约 25% 的 QBER，协议将判定
            窃听并中止。
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
    @staticmethod
    def _prepare(bit: int, basis: int) -> QCircuit:
        """Alice 制备第 0 个量子比特：bit=1 则 X，X 基则再加 H。"""
        qc = QCircuit()
        if bit:
            qc << X(0)
        if basis == QKD.BASIS_X:
            qc << H(0)
        return qc

    @staticmethod
    def _bob_basis_change(qc: QCircuit, basis: int) -> QCircuit:
        """Bob 在 Z 基测量前若选了 X 基，先补一个 H 把 |±> 变回 |0>/|1>。"""
        if basis == QKD.BASIS_X:
            qc << H(0)
        return qc

    def _round(self, bit: int, a_basis: int, b_basis: int) -> int:
        """运行单轮 BB84，返回 Bob 测出的比特（int）。"""
        if self.eve_prob > 0 and self.rng.random() < self.eve_prob:
            # Eve 拦截-重发：先以随机基窃听，再按自己测得的结果重新制备后发给 Bob
            e_basis = int(self.rng.integers(0, 2))
            qc_eve = self._prepare(bit, a_basis)
            if e_basis == QKD.BASIS_X:
                qc_eve << H(0)
            e_bit = int(self._sample_outcome(qc_eve, 1))
            # Eve 用 (e_bit, e_basis) 重新制备，再交给 Bob
            qc_bob = self._prepare(e_bit, e_basis)
            qc_bob = self._bob_basis_change(qc_bob, b_basis)
            return int(self._sample_outcome(qc_bob, 1))
        else:
            qc = self._prepare(bit, a_basis)
            qc = self._bob_basis_change(qc, b_basis)
            return int(self._sample_outcome(qc, 1))

    # ------------------------------- 密钥生成 ------------------------------- #
    def keygen(self, nlen: Optional[int] = None, verbose: bool = False) -> str:
        """
        运行 BB84 协议并返回最终密钥（二进制字符串）。

        参数
            nlen : int, optional
                最终密钥长度；省略时使用构造时的 key_len。
            verbose : bool
                为 True 时打印 QBER、sifted 长度等中间信息。
        返回
            key_str : str
                长度为 nlen 的二进制字符串。
        异常
            RuntimeError: 当估计的 QBER 超过安全阈值（疑似窃听）时抛出。
        """
        if nlen is None:
            nlen = self.key_len
        # 发送足够多的量子比特，确保 sifting + QBER 测试 + 隐私放大后仍有 nlen 位
        n_send = max(int(round(nlen * 8)), 128)

        alice_bits = []
        alice_bases = []
        bob_bits = []
        bob_bases = []
        for _ in range(n_send):
            a_bit = int(self.rng.integers(0, 2))
            a_basis = int(self.rng.integers(0, 2))
            b_basis = int(self.rng.integers(0, 2))
            b_bit = self._round(a_bit, a_basis, b_basis)
            alice_bits.append(a_bit)
            alice_bases.append(a_basis)
            bob_bits.append(b_bit)
            bob_bases.append(b_basis)

        # 1) sifting：保留双方基一致的轮次
        sift_a, sift_b = self._sift(alice_bases, bob_bases, alice_bits, bob_bits)
        if len(sift_a) == 0:
            raise RuntimeError("sifting 后没有保留任何比特，请增大 n_send 或 key_len")

        # 2) QBER 估计，超阈值则中止
        qber, keep = self._estimate_qber(sift_a, sift_b)
        if verbose:
            print("[BB84] sifted 长度=%d, 估计 QBER=%.3f, 阈值=%.3f"
                  % (len(sift_a), qber, self.qber_threshold))
        if not keep:
            raise RuntimeError(
                "检测到疑似窃听：QBER=%.3f 超过安全阈值 %.3f，协议中止"
                % (qber, self.qber_threshold)
            )

        # 3) 纠错近似：理想纠错后 sifted 双方一致，去掉 QBER 测试抽样位，
        #    用剩余比特做隐私放大
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

        # 4) 隐私放大
        final = self._privacy_amplification(remaining, final_len=nlen)
        return self._bits_to_str(final)
