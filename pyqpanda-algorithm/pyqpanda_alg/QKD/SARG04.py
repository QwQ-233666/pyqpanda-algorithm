# ~ref: https://en.wikipedia.org/wiki/SARG04
# ~ref: V. Scarani, A. Acín, G. Ribordy, N. Gisin, "Quantum cryptography
#       protocols robust against photon number splitting attacks",
#       Phys. Rev. Lett. 92, 057901 (2004).
"""
SARG04 量子密钥分发协议（Scarani-Acín-Ribordy-Gisin, 2004）。

SARG04 与 BB84 使用**完全相同的一组四个非正交态**来编码比特：

    Z 基：|0>            (a=0, b=0)      |1>           (a=0, b=1)
    X 基：|+> = H|0>     (a=1, b=0)      |-> = XH|0>   (a=1, b=1)

其中 a 为测量基（0=Z, 1=X），b 为比特值。也就是说，**同一个比特值 b 对应两个
跨基的非正交态**（b=0 对应 {|0>, |+>}，b=1 对应 {|1>, |->}）——这正是 SARG04
的「基矢编码特性」。

与 BB84 的关键差异（这正是它抗光子数分离攻击 PNS 的来源）：
  * BB84 中 Alice 在基比对阶段公开「基」；同基轮次 Bob 测得的比特即密钥比特。
  * SARG04 中 Alice 同样公开「基 a」，但**从不公开比特 b**。Bob 利用「非正交态
    之间的正交性」来推断 b：只有当 Bob 的测量结果正交于 Alice 所宣布基中某一个
    态时，他才能无歧义地反推出 b。在理想单光子模型下，这等价于「双方选了相同
    基」的轮次才可靠（不同基的测量结果对 b 只有 50% 正确率，须丢弃）；因此其
    筛选（sifting）效率与 BB84 相同，但凭借「仅公开基、不公开比特」的编码方式，
    在弱相干光（多光子）场景下对 PNS 攻击具有更强的健壮性。

安全性检验：与 BB84 类似，从同基筛选出的密钥中公开牺牲一小部分估计 QBER，
超阈值则中止。

本实现用 pyqpanda3 CPU 模拟器真实搭建「制备 -> 测量」单比特线路，通过后处理
（sifting / QBER / 隐私放大）得到最终密钥，并支持插入 Eve 的拦截-重发攻击。
"""

from typing import Optional

from .QKD import QKD
from pyqpanda3.core import QCircuit, X, H
import numpy as np


class SARG04(QKD):
    """
    SARG04 协议实现（基于 4 个非正交态的基矢编码，抗 PNS）。

    Parameters
        key_len : int
            期望最终密钥长度（比特）。
        qber_threshold : float
            QBER 安全阈值，默认 0.11（与 BB84 一致的安全上限）。
        eve_prob : float
            Eve 进行拦截-重发攻击的概率（0~1）。0 表示无窃听；非 0 时由于
            拦截-重发会引入约 25% 的 QBER，协议将判定窃听并中止。
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
        """
        Alice 制备第 0 个量子比特 |ψ_{basis,bit}>：
            basis=Z: bit=0 -> |0>,  bit=1 -> |1>      (先不施加 H)
            basis=X: bit=0 -> |+>,  bit=1 -> |->      (先 H 再按 bit 决定是否 X)
        """
        # 注意门顺序：必须「先 X 翻转、再 H 变基」。若先 H 后 X，则 X|+> = |+>
        # 会制备出错（bit=1 时得到 |+> 而非 |->）。正确的四种态：
        #   Z,0 -> |0>   Z,1 -> |1>      X,0 -> |+>   X,1 -> |-> = XH|0>
        qc = QCircuit()
        if bit:
            qc << X(0)
        if basis == QKD.BASIS_X:
            qc << H(0)
        return qc

    @staticmethod
    def _bob_derive_bit(a_basis: int, b_basis: int, r: int):
        """
        SARG04 中 Bob 根据「Alice 公开的基 a_basis」与自己的「(测量基, 结果)」
        反推比特 b，并判断该推断是否可靠。

        可靠性：仅当 Bob 的测量基与 Alice 的基相同（b_basis == a_basis）时，
        测量结果 r 才与比特 b 一一对应（r == b）；此时推断可靠。不同基下 r 与
        b 仅 50% 相关，推断不可靠，须丢弃。

        返回
            (bit, reliable) : Bob 推断的比特，以及该推断是否可靠。
        """
        reliable = (b_basis == a_basis)
        # 同基下：Z 基 r=0/1 对应 |0>/|1>；X 基 r=0/1 对应 |+>/|->，均对应 b=r
        return r, reliable

    def _round(self, a_bit: int, a_basis: int, b_basis: int, eve: bool = False) -> int:
        """
        运行单轮 SARG04，返回 Bob 的测量结果 r（0/1）。

        参数
            a_bit, a_basis : Alice 发送的比特与基。
            b_basis        : Bob 选择的测量基。
            eve            : 是否由 Eve 进行拦截-重发。
        返回
            r              : Bob 的测量结果（0/1）。
        """
        if eve:
            # Eve 拦截：以随机基窃听，再按她认为的（基, 比特）重新制备后发给 Bob
            e_basis = int(self.rng.integers(0, 2))
            qc_eve = self._prepare(a_bit, a_basis)
            if e_basis == QKD.BASIS_X:
                qc_eve << H(0)
            e_r = int(self._sample_outcome(qc_eve, 1))
            # Eve 把 (e_basis, e_r) 当作她测到的（基, 比特）重新制备
            qc = self._prepare(e_r, e_basis)
        else:
            qc = self._prepare(a_bit, a_basis)

        if b_basis == QKD.BASIS_X:
            qc << H(0)
        return int(self._sample_outcome(qc, 1))

    # ------------------------------- 密钥生成 ------------------------------- #
    def keygen(self, nlen: Optional[int] = None, verbose: bool = False) -> str:
        """
        运行 SARG04 协议并返回最终密钥（二进制字符串）。

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

        alice_bits = []   # 同基轮次上 Alice 的比特（即密钥比特）
        bob_bits = []     # Bob 在同基轮次测得的、与 Alice 一致的比特

        for _ in range(n_send):
            a_bit = int(self.rng.integers(0, 2))
            a_basis = int(self.rng.integers(0, 2))
            b_basis = int(self.rng.integers(0, 2))
            eve = self.eve_prob > 0 and self.rng.random() < self.eve_prob
            r = self._round(a_bit, a_basis, b_basis, eve=eve)

            # SARG04 sifting：Alice 公开基 a_basis 后，Bob 仅在「同基」轮次得到
            # 可靠结论（见 _bob_derive_bit）。同基下 Bob 的结果 r 即等于 Alice 比特。
            b_bit, reliable = self._bob_derive_bit(a_basis, b_basis, r)
            if reliable:
                alice_bits.append(a_bit)
                bob_bits.append(b_bit)

        sift_a = np.asarray(alice_bits, dtype=int)
        sift_b = np.asarray(bob_bits, dtype=int)
        if len(sift_a) == 0:
            raise RuntimeError("sifting 后没有保留任何比特，请增大 n_send 或 key_len")

        # QBER 估计：理想情况下同基轮次双方完全一致（QBER=0）
        qber, keep = self._estimate_qber(sift_a, sift_b)
        if verbose:
            print("[SARG04] 同基（可靠）轮次=%d, 估计 QBER=%.3f, 阈值=%.3f"
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
