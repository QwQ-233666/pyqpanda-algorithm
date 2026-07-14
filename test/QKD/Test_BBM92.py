#!/usr/bin/env python3
"""BBM92 协议功能测试（EPR 纠缠对 |Φ+>，BB84 的纠缠版本）。"""

import pytest
import warnings

from pyqpanda_alg.QKD import BBM92


class Test_QKD_BBM92:
    """BBM92 协议功能测试（EPR 纠缠对 |Φ+>，BB84 的纠缠版本）。"""

    def setup_method(self):
        warnings.filterwarnings("ignore")

    def test_bbm92_keygen_length_and_binary(self):
        key = BBM92(key_len=32, seed=2).keygen()
        assert isinstance(key, str)
        assert len(key) == 32
        assert all(c in "01" for c in key)

    def test_bbm92_reproducible_with_seed(self):
        k1 = BBM92(key_len=32, seed=321).keygen()
        k2 = BBM92(key_len=32, seed=321).keygen()
        assert k1 == k2, "相同随机种子应得到相同密钥"

    def test_bbm92_eve_detected(self):
        # Eve 拦截 Bob 一路粒子，纠缠被破坏，QBER 校验应触发中止
        with pytest.raises(RuntimeError):
            BBM92(key_len=32, eve_prob=1.0, seed=7).keygen()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
