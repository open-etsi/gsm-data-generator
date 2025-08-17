# tests/test_data_generator.py
import re
import pytest

import gsm_data_generator.generator as generator


# -------------------------
# Tests for DataGenerator
# -------------------------
def test_generate_ki():
    ki = generator.DataGenerator.generate_ki()
    assert isinstance(ki, str)
    assert len(ki) == 32  # 16 bytes -> 32 hex chars
    assert re.fullmatch(r"[0-9A-F]+", ki)


def test_generate_otas():
    otas = generator.DataGenerator.generate_otas()
    assert isinstance(otas, str)
    assert len(otas) == 32
    assert re.fullmatch(r"[0-9A-F]+", otas)


def test_generate_k4():
    k4 = generator.DataGenerator.generate_k4()
    assert isinstance(k4, str)
    assert len(k4) == 64  # 32 bytes -> 64 hex chars
    assert re.fullmatch(r"[0-9A-F]+", k4)


def test_generate_8_digit():
    num = generator.DataGenerator.generate_8_digit()
    assert num.isdigit()
    assert len(num) == 8


def test_generate_4_digit():
    num = generator.DataGenerator.generate_4_digit()
    assert num.isdigit()
    assert len(num) == 4


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
