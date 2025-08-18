# tests/test_data_generator.py
import re
import pytest

from gsm_data_generator.algorithm import CryptoUtils, DependentDataGenerator
from gsm_data_generator.algorithm import EncodingUtils
from gsm_data_generator.transform import DataTransform


# -------------------------
# Tests for DependentDataGenerator
# -------------------------
def test_calculate_opc(monkeypatch):
    def fake_calc_opc_hex(ki, op):
        return "a1b2c3d4"

    monkeypatch.setattr(CryptoUtils, "calc_opc_hex", fake_calc_opc_hex)

    opc = DependentDataGenerator.calculate_opc("11" * 16, "22" * 16)
    assert opc == "A1B2C3D4"


def test_calculate_eki(monkeypatch):
    def fake_aes_encrypt(transport, ki):
        return "deadbeef"

    monkeypatch.setattr(CryptoUtils, "aes_128_cbc_encrypt", fake_aes_encrypt)

    eki = DependentDataGenerator.calculate_eki("aa" * 16, "bb" * 16)
    assert eki == "deadbeef"


def test_calculate_acc():
    imsi = "123456789012345"
    acc = DependentDataGenerator.calculate_acc(imsi)
    assert isinstance(acc, str)
    assert len(acc) == 4
    assert re.fullmatch(r"[0-9a-f]+", acc)


# -------------------------
# Tests for DependentDataGenerator
# -------------------------
# def test_calculate_opc(monkeypatch):
#     def fake_calc_opc_hex(ki, op):
#         return "a1b2c3d4"

#     monkeypatch.setattr(CryptoUtils, "calc_opc_hex", fake_calc_opc_hex)

#     opc = DependentDataGenerator.calculate_opc("11" * 16, "22" * 16)
#     assert opc == "A1B2C3D4"


# def test_calculate_eki(monkeypatch):
#     def fake_aes_encrypt(transport, ki):
#         return "deadbeef"

#     monkeypatch.setattr(CryptoUtils, "aes_128_cbc_encrypt", fake_aes_encrypt)

#     eki = DependentDataGenerator.calculate_eki("aa" * 16, "bb" * 16)
#     assert eki == "deadbeef"


# def test_calculate_acc():
#     imsi = "123456789012345"
#     acc = DependentDataGenerator.calculate_acc(imsi)
#     assert isinstance(acc, str)
#     assert len(acc) == 4
#     assert re.fullmatch(r"[0-9a-f]+", acc)
