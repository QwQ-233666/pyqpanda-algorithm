#!/usr/bin/env python3
"""E91 协议功能测试（纠缠 + CHSH 校验）。"""

import pytest
import warnings

from pyqpanda_alg.QKD import E91


class Test_QKD_E91:
    """E91 协议功能测试（纠缠 + CHSH 校验）。"""

    def setup_method(self):
        warnings.filterwarnings("ignore")

    def test_e91_keygen_length_and_binary(self):
        key = E91(key_len=32, seed=2).keygen()
        assert isinstance(key, str)
        assert len(key) == 32
        assert all(c in "01" for c in key)

    def test_e91_reproducible_with_seed(self):
        k1 = E91(key_len=32, seed=321).keygen()
        k2 = E91(key_len=32, seed=321).keygen()
        assert k1 == k2, "相同随机种子应得到相同密钥"

    def test_e91_eve_detected(self):
        # Eve 拦截 Bob 一路粒子，纠缠被破坏，CHSH/QBER 校验应触发中止
        with pytest.raises(RuntimeError):
            E91(key_len=32, eve_prob=1.0, seed=7).keygen()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
