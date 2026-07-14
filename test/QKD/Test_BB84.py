#!/usr/bin/env python3
"""BB84 协议功能测试。"""

import pytest
import warnings

from pyqpanda_alg.QKD import BB84


class Test_QKD_BB84:
    """BB84 协议功能测试。"""

    def setup_method(self):
        warnings.filterwarnings("ignore")

    def test_bb84_keygen_length_and_binary(self):
        key = BB84(key_len=32, seed=1).keygen()
        assert isinstance(key, str)
        assert len(key) == 32
        assert all(c in "01" for c in key)

    def test_bb84_reproducible_with_seed(self):
        k1 = BB84(key_len=32, seed=123).keygen()
        k2 = BB84(key_len=32, seed=123).keygen()
        assert k1 == k2, "相同随机种子应得到相同密钥"

    def test_bb84_eve_detected(self):
        # Eve 进行 100% 拦截-重发，QBER 应显著升高，协议中止
        with pytest.raises(RuntimeError):
            BB84(key_len=32, eve_prob=1.0, seed=7).keygen()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
