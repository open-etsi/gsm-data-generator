# tests/test_data_generator.py
import re
import pytest

import gsm_data_generator.generator as generator


# -------------------------
# Tests for DependentDataGenerator
# -------------------------
def test_calculate_opc(monkeypatch):
    def fake_calc_opc_hex(ki, op):
        return "a1b2c3d4"

    monkeypatch.setattr(generator.CryptoUtils, "calc_opc_hex", fake_calc_opc_hex)

    opc = generator.DependentDataGenerator.calculate_opc("11" * 16, "22" * 16)
    assert opc == "A1B2C3D4"


def test_calculate_eki(monkeypatch):
    def fake_aes_encrypt(transport, ki):
        return "deadbeef"

    monkeypatch.setattr(generator.CryptoUtils, "aes_128_cbc_encrypt", fake_aes_encrypt)

    eki = generator.DependentDataGenerator.calculate_eki("aa" * 16, "bb" * 16)
    assert eki == "deadbeef"


def test_calculate_acc():
    imsi = "123456789012345"
    acc = generator.DependentDataGenerator.calculate_acc(imsi)
    assert isinstance(acc, str)
    assert len(acc) == 4
    assert re.fullmatch(r"[0-9a-f]+", acc)
