# ~ref: https://en.wikipedia.org/wiki/B92_protocol
# ~ref: C. H. Bennett, "Quantum cryptography using any two nonorthogonal states",
#       Phys. Rev. Lett. 68, 3121 (1992).
"""
B92 量子密钥分发协议（Bennett, 1992）。

B92 是 BB84 的「极简」版本，只用 **两个非正交单光子态** 编码一个比特：
    * 比特 0  ->  |0>  （Z 基的 |0>）
    * 比特 1  ->  |+> = (|0> + |1>)/√2  （X 基的 |+>）

与 BB84（每个基里挑一个比特）不同，B92 里 Alice 发的态由「比特值」直接决定，
而 **测量基由 Bob 随机选**（Z 或 X），不再需要 Alice 也随机选基。

核心物理：两个态 |0> 与 |+> 非正交（|<0|+>|² = 1/2），因此 Bob 只有「部分」
测量结果能**无歧义**地反推出 Alice 发的到底是哪一个态；其余结果是「非结论性」
（inconclusive）的，必须丢弃。这恰是 B92 的安全来源——窃听者同样无法对任意
结果做出确定判断。

Bob 的测量与「结论性」判定（用 Z/X 基测量，结果记为 r∈{0,1}，其中 X 基下
r=1 对应 |-> 态）：
    * 选 Z 基且测得 r=1（即 |1>）：|1> 与 |0> 正交 ⇒ 态必为 |+> ⇒ Alice 比特=1。
    * 选 X 基且测得 r=1（即 |->）：|-> 与 |+> 正交 ⇒ 态必为 |0> ⇒ Alice 比特=0。
    * 其余结果（r=0）非结论性，丢弃。
于是「结论性轮次」只占约 1/4（每种比特各 1/4），但其携带的比特是确定且平衡的。

本实现用 pyqpanda3 CPU 模拟器真实搭建「制备 -> 测量」单比特线路，通过后处理
（sifting / QBER / 隐私放大）得到最终密钥，并支持插入 Eve 的拦截-重发攻击以
演示窃听检测。
"""

from typing import Optional

from .QKD import QKD
from pyqpanda3.core import QCircuit, X, H
import numpy as np


class B92(QKD):
    """
    B92 协议实现（基于两个非正交态 |0> 与 |+>）。

    Parameters
        key_len : int
            期望最终密钥长度（比特）。
        qber_threshold : float
            QBER 安全阈值，默认 0.11（与 BB84 一致的安全上限）。
        eve_prob : float
            Eve 进行拦截-重发攻击的概率（0~1）。0 表示无窃听；非 0 时由于
            拦截-重发会引入误码，使 QBER 升高、协议中止。
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
    def _prepare_alice(bit: int) -> QCircuit:
        """
        Alice 制备第 0 个量子比特：
            bit=0 -> |0>   （不施加任何门）
            bit=1 -> |+> = H|0>
        """
        qc = QCircuit()
        if bit:
            qc << H(0)
        return qc

    def _round(self, bit: int, b_basis: int, eve: bool = False) -> int:
        """
        运行单轮 B92，返回 Bob 的测量结果 r（0 或 1）。

        参数
            bit     : Alice 发送的比特（0 -> |0>, 1 -> |+>）。
            b_basis : Bob 选择的测量基（0=Z, 1=X）。
            eve     : 是否由 Eve 进行拦截-重发。
        返回
            r       : Bob 的测量结果（0/1）。
        """
        # 1) Alice 制备
        qc = self._prepare_alice(bit)

        # 2) 可选：Eve 拦截-重发
        if eve:
            e_basis = int(self.rng.integers(0, 2))
            qc_eve = self._prepare_alice(bit)
            if e_basis == QKD.BASIS_X:
                qc_eve << H(0)
            e_r = int(self._sample_outcome(qc_eve, 1))
            # Eve 依据自己的（基, 结果）重新制备一个态再发给 Bob
            if e_basis == QKD.BASIS_Z:
                e_state_bit = e_r             # Z 下 r=0->|0>(bit0), r=1->|+>(bit1)
            else:                             # X 下 r=0->|+>(bit1), r=1->|0>(bit0)
                e_state_bit = 1 - e_r
            qc = self._prepare_alice(e_state_bit)

        # 3) Bob 测量：X 基先补一个 H 把 |±> 变回 |0>/|1> 再测 Z
        if b_basis == QKD.BASIS_X:
            qc << H(0)
        return int(self._sample_outcome(qc, 1))

    # ------------------------------- 密钥生成 ------------------------------- #
    def keygen(self, nlen: Optional[int] = None, verbose: bool = False) -> str:
        """
        运行 B92 协议并返回最终密钥（二进制字符串）。

        参数
            nlen   : int, optional
                最终密钥长度；省略时使用构造时的 key_len。
            verbose: bool
                为 True 时打印结论性轮次数、QBER 等中间信息。
        返回
            key_str: str
                长度为 nlen 的二进制字符串。
        异常
            RuntimeError: 当估计的 QBER 超过安全阈值（疑似窃听）时抛出。
        """
        if nlen is None:
            nlen = self.key_len
        # 结论性轮次约占 1/4，故发送量需明显大于最终密钥长度
        n_send = max(int(round(nlen * 16)), 256)

        alice_bits = []   # 结论性轮次上 Alice 的原始比特（即密钥比特）
        bob_bits = []     # Bob 由（基, 结果）推出的比特

        for _ in range(n_send):
            a_bit = int(self.rng.integers(0, 2))
            b_basis = int(self.rng.integers(0, 2))
            eve = self.eve_prob > 0 and self.rng.random() < self.eve_prob
            r = self._round(a_bit, b_basis, eve=eve)

            # 结论性判定（见模块 docstring）
            conclusive = (b_basis == QKD.BASIS_Z and r == 1) or \
                         (b_basis == QKD.BASIS_X and r == 1)
            if conclusive:
                # Z 基测到 1 -> Alice 发的是 |+> -> 比特 1
                # X 基测到 1（即 |->）-> Alice 发的是 |0> -> 比特 0
                b_bit = 1 if (b_basis == QKD.BASIS_Z and r == 1) else 0
                alice_bits.append(a_bit)
                bob_bits.append(b_bit)

        sift_a = np.asarray(alice_bits, dtype=int)
        sift_b = np.asarray(bob_bits, dtype=int)
        if len(sift_a) == 0:
            raise RuntimeError("没有任何结论性轮次，请增大 n_send 或 key_len")

        # QBER 估计：理想情况下 sift_a == sift_b（QBER=0）
        qber, keep = self._estimate_qber(sift_a, sift_b)
        if verbose:
            print("[B92] 结论性轮次=%d, 估计 QBER=%.3f, 阈值=%.3f"
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
