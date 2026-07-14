#!/usr/bin/env python3
"""SARG04 协议功能测试（4 非正交态 + 基矢编码，抗 PNS）。"""

import pytest
import warnings

from pyqpanda_alg.QKD import SARG04


class Test_QKD_SARG04:
    """SARG04 协议功能测试（4 非正交态 + 基矢编码，抗 PNS）。"""

    def setup_method(self):
        warnings.filterwarnings("ignore")

    def test_sarg04_keygen_length_and_binary(self):
        key = SARG04(key_len=32, seed=3).keygen()
        assert isinstance(key, str)
        assert len(key) == 32
        assert all(c in "01" for c in key)

    def test_sarg04_reproducible_with_seed(self):
        k1 = SARG04(key_len=32, seed=333).keygen()
        k2 = SARG04(key_len=32, seed=333).keygen()
        assert k1 == k2, "相同随机种子应得到相同密钥"

    def test_sarg04_eve_detected(self):
        # Eve 进行 100% 拦截-重发，QBER 应显著升高，协议中止
        with pytest.raises(RuntimeError):
            SARG04(key_len=32, eve_prob=1.0, seed=7).keygen()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
