# ~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution#E91_protocol:_Artur_Ekert_.281991.29
"""
E91 量子密钥分发协议（Ekert, 1991）。

E91 是首个基于量子纠缠的 QKD 协议：
  * 一个可信（或半可信）源为每一轮分发一个最大纠缠 Bell 对
    |Φ+> = (|00> + |11>)/√2，分别发给 Alice 与 Bob；
  * 双方各自**独立随机**选择测量基（这里用 4 个方向：Z、X、π/4、3π/4，
    以绕 Y 轴旋转 RY(θ) 后测量 Z 来实现）；
  * **密钥提取**：当双方都选择了 Z 或都选择了 X 时，测量结果正相关
    （Bob 的比特与 Alice 一致），用 Alice 的比特作为原始密钥；
  * **安全性检验**：用「基不同」的那部分轮次计算 CHSH 不等式的 S 值。
    若源是最大纠缠的，理想情况下 |S| = 2√2 ≈ 2.828 > 2（量子力学允许的上界）；
    一旦出现窃听或退相干，纠缠被破坏，|S| 会跌到经典上界 2，协议中止。

本实现用 pyqpanda3 的 CPU 模拟器真正搭建 Bell 态与测量线路，通过线路
概率分布采样得到测量比特；CHSH 估计与隐私放大复用 `QKD` 基类。可选插入
Eve 的拦截-重发攻击以演示 S 值下降、协议中止。
"""

from typing import Optional

import numpy as np
from .QKD import QKD
from pyqpanda3.core import QCircuit, H, CNOT, RY, X


# 4 个测量方向（绕 Y 轴的角度 θ；实际线路中对相应比特施加 RY(θ) 再测 Z）。
# 对 |Φ+>，测量关联 E(θ_A, θ_B) = cos(θ_A - θ_B)。
#   0 -> Z 基 (0)        1 -> X 基 (π/2)
#   2 -> π/4             3 -> 3π/4
# 密钥提取用 {0,1}（Z、X，均正相关），CHSH 用 Alice{0,1} 与 Bob{2,3}
# 的组合达到最大违背 |S| = 2√2 ≈ 2.828。
ANGLES = [0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4]

# 用于 CHSH 的组合：Alice 取 {0,1}，Bob 取 {2,3}
#   S = E(0,2) - E(0,3) + E(1,2) + E(1,3)  ->  |S| = 2√2
_CHSH_COMBOS = [(0, 2), (0, 3), (1, 2), (1, 3)]
# 密钥轮次：双方都选 Z (0,0) 或都选 X (1,1)
_KEY_COMBOS = [(0, 0), (1, 1)]


class E91(QKD):
    """
    E91 协议实现。

    Parameters
        key_len : int
            期望最终密钥长度（比特）。
        qber_threshold : float
            QBER 安全阈值（作为额外校验），默认 0.11。
        chsh_threshold : float
            CHSH 经典上界，默认 2.0；|S| 低于该值即判定不安全并中止。
        eve_prob : float
            Eve 对 Bob 那一路粒子做拦截-重发攻击的概率（0~1）。
            非 0 时会破坏纠缠、压低 |S| 并抬升 QBER，使协议中止。
        seed : int or None
            随机种子，便于复现。
    """

    def __init__(self, key_len: int = 64, qber_threshold: float = 0.11,
                 chsh_threshold: float = 2.0, eve_prob: float = 0.0,
                 seed: int = None):
        super().__init__(key_len=key_len, qber_threshold=qber_threshold, seed=seed)
        if chsh_threshold <= 0:
            raise ValueError("chsh_threshold 必须为正数")
        if not (0.0 <= eve_prob <= 1.0):
            raise ValueError("eve_prob 应位于 [0, 1]")
        self.chsh_threshold = float(chsh_threshold)
        self.eve_prob = float(eve_prob)

    # ----------------------------- 量子线路工具 ----------------------------- #
    def _bell(self) -> QCircuit:
        """制备 Bell 态 |Φ+> = (|00>+|11>)/√2。"""
        qc = QCircuit()
        qc << H(0) << CNOT(0, 1)
        return qc

    def _measure_bell(self, a_setting: int, b_setting: int, shots: int = 1024):
        """对 Bell 源的两个粒子分别在 θ_a、θ_b 方向测量，返回 (a_bit, b_bit)。"""
        qc = self._bell()
        qc << RY(0, ANGLES[a_setting])
        qc << RY(1, ANGLES[b_setting])
        out = self._sample_outcome(qc, 2, shots=shots)
        return int(out[0]), int(out[1])

    def _measure_bell_eve(self, a_setting: int, b_setting: int, e_setting: int,
                          shots: int = 1024):
        """
        Eve 拦截 Bob 那一路：先测自己那半得到 e_bit，再用 (e_bit, e_setting)
        重新制备一个粒子发给 Bob，Bob 再按 θ_b 测量。返回 (a_bit, b_bit)。
        """
        # 1) Alice 测自己那半（θ_a），同时 Eve 测原 Bob 那半（θ_e）
        qc1 = self._bell()
        qc1 << RY(0, ANGLES[a_setting])
        qc1 << RY(1, ANGLES[e_setting])
        out1 = self._sample_outcome(qc1, 2, shots=shots)
        a_bit = int(out1[0])
        e_bit = int(out1[1])
        # 2) Eve 重新制备 (e_bit, e_setting) 交给 Bob，Bob 按 θ_b 测量
        qc2 = QCircuit()
        if e_bit:
            qc2 << X(0)
        qc2 << RY(0, ANGLES[e_setting])   # Eve 在她测得的基上重新制备
        qc2 << RY(0, ANGLES[b_setting])   # Bob 在自己的基上测量
        b_bit = int(self._sample_outcome(qc2, 1, shots=shots))
        return a_bit, b_bit

    # ------------------------------- 密钥生成 ------------------------------- #
    def keygen(self, nlen: Optional[int] = None, verbose: bool = False) -> str:
        """
        运行 E91 协议并返回最终密钥（二进制字符串）。

        参数
            nlen : int, optional
                最终密钥长度；省略时使用构造时的 key_len。
            verbose : bool
                为 True 时打印 CHSH 的 S 值与 QBER 等中间信息。
        返回
            key_str : str
                长度为 nlen 的二进制字符串。
        异常
            RuntimeError: 当 |CHSH S| 不低于经典上界（纠缠被破坏/疑似窃听）
                或 QBER 超过阈值时抛出。
        """
        if nlen is None:
            nlen = self.key_len
        n_send = max(int(round(nlen * 24)), 1024)

        raw_key = []                 # 密钥轮次中 Alice 的比特
        key_settings = []            # 密钥轮次对应的 Alice 基 (0=Z, 1=X)
        key_bob = []                 # 密钥轮次中 Bob 实际测得的比特
        chsh_samples = {c: [] for c in _CHSH_COMBOS}

        for _ in range(n_send):
            a = int(self.rng.integers(0, 4))
            b = int(self.rng.integers(0, 4))

            if self.eve_prob > 0 and self.rng.random() < self.eve_prob:
                e = int(self.rng.integers(0, 4))
                a_bit, b_bit = self._measure_bell_eve(a, b, e)
            else:
                a_bit, b_bit = self._measure_bell(a, b)

            if (a, b) in _KEY_COMBOS:
                raw_key.append(a_bit)
                key_settings.append(a)
                key_bob.append(b_bit)
            elif (a, b) in _CHSH_COMBOS:
                chsh_samples[(a, b)].append((2 * a_bit - 1, 2 * b_bit - 1))

        # 1) CHSH 检验：|S| 跌破经典上界则中止
        Es = {}
        for c in _CHSH_COMBOS:
            vals = chsh_samples[c]
            Es[c] = float(np.mean([x * y for x, y in vals])) if vals else 0.0
        S = Es[(0, 2)] - Es[(0, 3)] + Es[(1, 2)] + Es[(1, 3)]
        if verbose:
            print("[E91] CHSH S=%.3f (经典上界=%.2f, 量子上界 2√2≈2.828)"
                  % (S, self.chsh_threshold))
        if abs(S) <= self.chsh_threshold:
            raise RuntimeError(
                "纠缠被破坏：|CHSH S|=%.3f <= %.3f，疑似窃听，协议中止"
                % (abs(S), self.chsh_threshold)
            )

        # 2) QBER 估计（密钥轮次）：对 |Φ+>，匹配基下双方正相关，
        #    故 QBER = Alice 与 Bob 比特不一致的比例，超阈值则中止
        if key_settings:
            errors = sum(1 for i in range(len(raw_key)) if raw_key[i] != key_bob[i])
            qber = errors / len(key_settings)
        else:
            qber = 1.0
        if verbose:
            print("[E91] 密钥轮次 QBER=%.3f, 原始密钥长度=%d"
                  % (qber, len(raw_key)))
        if qber > self.qber_threshold:
            raise RuntimeError(
                "QBER=%.3f 超过安全阈值 %.3f，协议中止" % (qber, self.qber_threshold)
            )

        # 3) 隐私放大：把原始密钥压缩为最终密钥
        if len(raw_key) < nlen:
            raise RuntimeError(
                "原始密钥长度不足（%d < %d），请增大 key_len 或 n_send"
                % (len(raw_key), nlen)
            )
        final = self._privacy_amplification(np.asarray(raw_key, dtype=int),
                                            final_len=nlen)
        return self._bits_to_str(final)
