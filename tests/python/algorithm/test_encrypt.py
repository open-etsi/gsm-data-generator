# tests/test_data_generator.py
import re
import pytest

from gsm_data_generator.algorithm import CryptoUtils, DependentDataGenerator
from gsm_data_generator.algorithm import EncodingUtils
from gsm_data_generator.transform import DataTransform


import re
import pytest
from gsm_data_generator.algorithm import DependentDataGenerator


@pytest.mark.parametrize("imsi", [
    "123456789012345",
    "310150123456789",
    "460001234567890",
    "724999123456789",
    "262071234567890",
    "505021234567890",
    "208201234567890",
    "440101234567890",
    "234151234567890",
    "404451234567890",
])
def test_calculate_acc(imsi):
    acc = DependentDataGenerator.calculate_acc(imsi)

    assert isinstance(acc, str), f"ACC must be str for IMSI {imsi}"
    assert len(acc) == 4, f"ACC must be 4 hex chars for IMSI {imsi}"
    assert re.fullmatch(r"[0-9a-f]+", acc), f"ACC must be lowercase hex for IMSI {imsi}: {acc}"

# def test_calculate_acc():
#     imsi = "123456789012345"
#     acc = DependentDataGenerator.calculate_acc(imsi)
#     assert isinstance(acc, str)
#     assert len(acc) == 4
#     assert re.fullmatch(r"[0-9a-f]+", acc)


test_vectors = [
    (
        "7762AD5BFABE4E189A06AA534C27C28F",
        "B9AFD4DEB8DABD4979DF485AD9B5CCD2",
        "3CE7B58AFA24D28EF644397DA92ECE36",
        "5C9E1D11B4923D85BA08D333EABF7E2D",
        "E881D500613657A970AC7FECCF09ACF2",
    ),
    (
        "1DAD3010782F403DB3355AC40ABE9343",
        "F84430F67A129591CCAF40AB44A3C357",
        "9898966C02C7EE70FC9E8C807F37D442",
        "79925E138AEBF5D65C6C49AAD6592F19",
        "800708A4B569DF12C3A5427084883392",
    ),
    (
        "111682ABDBEE4618B300C77142BB34E3",
        "BF7B6313533A12C88E39FAA58F872E7A",
        "E9D66BD05C3E24FD4978FEFAD0EEE21E",
        "55CF2BB76463A92934E12D7516E5226E",
        "81AE1365374BF2E38AE8C6DFBDA84FCC",
    ),
    (
        "FD231C0D256A4CA0B35A615C711F366C",
        "2FE1E7B0CCBBC584421C0DDAB85B6B88",
        "72B2CD05052A6B4C26C3B3B98AC28912",
        "B5836881A2E125F4B637ADDDED4717D4",
        "C89373B03BA0193F95E86211C0BFEDCF",
    ),
    (
        "2D30811A1E74484688A7AFC2D818D553",
        "3DC86F6D09B8B64247206378F451AB3A",
        "4A646085A96B49E3A17B72E24FF46350",
        "92778989652D31870EF0C2526BB1BB2D",
        "101C7A3FC923395E959A940E6A57800E",
    ),
    (
        "51FFDFB2F15C4DB7A080C6C1781792D5",
        "9583BD70178705BED2BAC239ECDEBADC",
        "6FC169107CEDEC81E02151104055D081",
        "927F89F2CB8FFE1F4DE3D13BAA569941",
        "1EDD0600EA018FF809D6BB5FDD583969",
    ),
    (
        "549330CC1FCA4A488E4D1F34FC335ECC",
        "EB015B739EFF6C086618A798EB9A36F5",
        "BC583EA39F08C1483DA73EEC9272C0D3",
        "13F117C6C17CA8EEDEAEB297DC5A570C",
        "D9045A914EDF61210B94EC166E8D9A94",
    ),
    (
        "13D098496569471B838AB9DF8DB014A1",
        "0CB016549813AEB2B8FAB02512A28734",
        "F92A3310296E47FFD2A2EF2B8003335C",
        "1C0E4272F30770A620C6A78592455E34",
        "3ADD69B191A5BB0A61BBBA4AE04357B0",
    ),
    (
        "ADE93B3694F94F7BAC544288F1B4B4B2",
        "48DF6181A473B15D8701C7647E84B3CE",
        "2CC2B1F032A0C9F7EF12BBE111548FBC",
        "C4386B8F1E7E39F4D37199ABCCBC233D",
        "5CB63204ADA1AE260887C8726A339456",
    ),
    (
        "9C315075A9C04293B9CFAC81A69251FF",
        "D76ADB268F379B519EBF2E4E06383033",
        "62331CEE1E35FCD1335A70C5D1382F90",
        "B5EEE82B33B54D245DEB74B5C75B796D",
        "10B128A09CFB056C819A3929CAED70D6",
    ),
]

# vectors.append((ki, op, transport, opc, eki))


@pytest.mark.parametrize("ki, op, transport, expected_opc, expected_eki", test_vectors)
def test_opc_and_eki_generation(ki, op, transport, expected_opc, expected_eki):
    """Validate OPC and eKI calculation with real known-good vectors."""

    opc = DependentDataGenerator.calculate_opc(op, ki)
    eki = DependentDataGenerator.calculate_eki(transport, ki)

    assert opc.upper() == expected_opc.upper()
    assert eki.upper() == expected_eki.upper()
