# ~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution
"""
Base class and shared classical post-processing for Quantum Key Distribution (QKD).

A QKD protocol produces a *raw key* on both sides (Alice and Bob) via a quantum
channel, then both parties run classical post-processing to turn it into a
secret key:

    1. Sifting        -- keep only the rounds where Alice & Bob used a
                         compatible basis (or, for E91, where their bases
                         happen to be correlated).
    2. Error estimate -- sample a subset of the sifted key to estimate the
                         Quantum Bit Error Rate (QBER). If QBER exceeds a
                         security threshold the protocol aborts (an eavesdropper
                         is suspected).
    3. Privacy ampl.  -- compress the corrected key with a random Toeplitz
                         matrix so that any partial information leaked to an
                         eavesdropper is washed out.

The quantum part (state preparation / measurement / entanglement) is delegated
to each concrete protocol subclass (BB84, E91, ...). This base class only
implements the classical parts plus a small helper to sample a measurement
outcome *from a quantum-circuit probability distribution*, so the protocols can
run on the pyqpanda3 CPU simulator.
"""

import numpy as np

from pyqpanda3.core import CPUQVM, QCircuit, QProg


_BASIS_Z = 0  # computational basis {|0>, |1>}
_BASIS_X = 1  # Hadamard / diagonal basis {|+>, |->}


class QKD:
    """
    量子密钥分发协议的公共基类。

    Parameters
        key_len : int
            期望最终密钥的**比特长度**（隐私放大之后的长度）。
        qber_threshold : float
            QBER 安全阈值；估计的误码率超过该值则判定疑似窃听并中止。
            标准 BB84 的安全上限约为 11%（0.11）。
        seed : int or None
            随机数种子，便于复现实验结果；为 None 时使用系统随机源。

    Attributes
        key_len, qber_threshold : 见上
        rng : numpy.random.Generator
            由 seed 派生的随机数生成器。
    """

    #: 比特/基选择的取值范围，子类可覆盖
    BASIS_Z = _BASIS_Z
    BASIS_X = _BASIS_X

    def __init__(self, key_len: int = 128, qber_threshold: float = 0.11, seed: int = None):
        if key_len <= 0:
            raise ValueError("key_len 必须为正整数")
        if not (0.0 < qber_threshold < 0.5):
            raise ValueError("qber_threshold 应位于 (0, 0.5) 之间")
        self.key_len = int(key_len)
        self.qber_threshold = float(qber_threshold)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 量子部分：交由子类实现
    # ------------------------------------------------------------------ #
    def keygen(self, nlen: int) -> str:
        """
        运行完整的 QKD 协议，返回最终密钥（二进制字符串）。

        子类必须覆写此方法，串联「量子分发 -> sifting -> QBER 估计 ->
        纠错（此处用理想纠错近似）-> 隐私放大」。
        """
        raise NotImplementedError("子类必须实现 keygen 方法")

    # ------------------------------------------------------------------ #
    # 经典后处理：公共工具方法
    # ------------------------------------------------------------------ #
    def _sample_outcome(self, qc: QCircuit, n_qubits: int, shots: int = 1024):
        """
        在 CPU 模拟器上运行一个线路，并根据其末态的概率分布采样一个测量结果。

        这里直接对线路末态调用 ``get_prob_dict``（与 ``QSVR`` 示例一致，无需
        显式测量门），按 Born 规则得到的基分布抽样，等价于做一次测量。

        参数
            qc       : 待运行的 QCircuit（不含测量）。
            n_qubits : 参与测量的量子比特个数，结果串长度为 n_qubits。
            shots    : 模拟的 shot 数（仅用于触发运行，分布由末态决定）。
        返回
            outcome  : 形如 "010" 的二进制字符串（高位在前，对应 qubit 0）。
        """
        machine = CPUQVM()
        prog = QProg(n_qubits)
        prog << qc
        machine.run(prog, shots)
        prob = machine.result().get_prob_dict(prog.qubits())

        outcomes = list(prob.keys())
        probs = np.array([prob[k] for k in outcomes], dtype=float)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(outcomes), p=probs)
        return outcomes[idx]

    def _sift(self, alice_bases, bob_bases, alice_bits, bob_bits=None):
        """
        筛选双方基一致的轮次。

        参数
            alice_bases, bob_bases : sequence of int
                双方的基选择（0/1）。长度必须相等。
            alice_bits, bob_bits   : sequence of int, optional
                Alice 发送的比特（必填）与 Bob 测出的比特（可选）。
                若省略 bob_bits 则只返回基一致的索引与 Alice 比特。
        返回
            sifted_a : np.ndarray
                Alice 在匹配轮次上的比特。
            sifted_b : np.ndarray or None
                Bob 在匹配轮次上的比特（bob_bits 给出时）；否则为 None。
        """
        ab = np.asarray(alice_bases)
        bb = np.asarray(bob_bases)
        if ab.shape != bb.shape:
            raise ValueError("alice_bases 与 bob_bases 长度必须相等")
        mask = ab == bb
        sifted_a = np.asarray(alice_bits)[mask]
        sifted_b = np.asarray(bob_bits)[mask] if bob_bits is not None else None
        return sifted_a, sifted_b

    def _estimate_qber(self, sifted_a, sifted_b, test_fraction: float = 0.2):
        """
        抽样估计 QBER，并据此判断是否中止协议。

        参数
            sifted_a, sifted_b : np.ndarray
                筛选后的双方比特串（长度必须相等）。
            test_fraction : float
                用于公开比对以估计误码率的样本比例（0,1]。
        返回
            (qber, keep) : (float, bool)
                qber 为估计的误码率；keep 为 True 表示误码率在安全阈值内、
                协议可继续，False 表示疑似窃听、应中止。
        """
        n = len(sifted_a)
        if n == 0:
            return 1.0, False
        k = max(1, int(round(n * test_fraction)))
        k = min(k, n)
        idx = self.rng.choice(n, size=k, replace=False)
        errors = int(np.sum(sifted_a[idx] != sifted_b[idx]))
        qber = errors / k
        keep = qber <= self.qber_threshold
        return qber, keep

    def _privacy_amplification(self, key_bits, final_len: int = None):
        """
        用随机 Toeplitz 矩阵对（已纠错的）密钥做隐私放大。

        Toeplitz 矩阵由 `2*final_len-1` 个随机二进制位唯一确定，可高效生成。

        参数
            key_bits : sequence of int
                纠错后的原始密钥比特（长度 >= final_len）。
            final_len : int, optional
                输出密钥长度；默认取构造时的 self.key_len。
        返回
            final_key : np.ndarray(dtype=int)
                压缩后的最终密钥比特。
        """
        if final_len is None:
            final_len = self.key_len
        key_bits = np.asarray(key_bits, dtype=int)
        n = len(key_bits)
        if final_len > n:
            raise ValueError("final_len 不能大于密钥长度 n=%d" % n)

        m = 2 * final_len - 1
        # first_row 与 first_col 唯一确定 Toeplitz 矩阵
        first_row = self.rng.integers(0, 2, size=final_len, dtype=int)
        first_col = self.rng.integers(0, 2, size=final_len, dtype=int)
        # 去掉重复的中心元素 t[0,0]
        toeplitz_vec = np.concatenate([first_col[::-1][:-1], first_row])  # 长度 2*final_len-1
        assert len(toeplitz_vec) == m

        out = np.zeros(final_len, dtype=int)
        for i in range(final_len):
            # 第 i 行取 key_bits[i : i + final_len]，与 Toeplitz 向量的第 i 行
            #（沿反对角线平移，故对切片取逆）做按位与后异或归约
            window = key_bits[i:i + final_len]
            row_vec = toeplitz_vec[i:i + final_len][::-1]
            out[i] = int(np.bitwise_xor.reduce(window & row_vec))
        return out

    # ------------------------------------------------------------------ #
    # 一些便于子类使用的随机/编码工具
    # ------------------------------------------------------------------ #
    def _random_bits(self, n: int):
        return self.rng.integers(0, 2, size=n, dtype=int)

    def _random_bases(self, n: int):
        return self.rng.integers(0, 2, size=n, dtype=int)

    @staticmethod
    def _bits_to_str(bits):
        return "".join(str(int(b)) for b in bits)
