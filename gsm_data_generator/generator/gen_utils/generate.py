import secrets
from .encrypt import CryptoUtils


class DependentDataGenerator:
    @staticmethod
    def calculate_opc(op: str, ki: str) -> str:
        return CryptoUtils.calc_opc_hex(ki, op).upper()

    @staticmethod
    def calculate_eki(transport: str, ki: str) -> str:
        return CryptoUtils.aes_128_cbc_encrypt(transport, ki)

    @staticmethod
    def calculate_acc(imsi: str) -> str:
        last_digit = int(imsi[-1])
        acc_binary = bin(1 << last_digit)[2:].zfill(16)
        return format(int(acc_binary, 2), "04x")


class DataGenerator:
    @staticmethod
    def generate_ki() -> str:
        return secrets.token_hex(16).upper()

    @staticmethod
    def generate_otas() -> str:
        return secrets.token_hex(16).upper()

    @staticmethod
    def generate_k4() -> str:
        return secrets.token_hex(32).upper()

    @staticmethod
    def generate_8_digit() -> str:
        return str(secrets.SystemRandom().randint(10000000, 99999999))

    @staticmethod
    def generate_4_digit() -> str:
        return str(secrets.SystemRandom().randint(1000, 9999))


__all__ = ["DependentDataGenerator", "DataGenerator"]
