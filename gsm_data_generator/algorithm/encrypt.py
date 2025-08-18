from Crypto.Cipher import AES
import binascii


class CryptoUtils:
    @staticmethod
    def aes_128_cbc_encrypt(key: str, text: str) -> str:
        iv = binascii.unhexlify("00000000000000000000000000000000")
        key_bytes = binascii.unhexlify(key)
        text_bytes = binascii.unhexlify(text)
        encryptor = AES.new(key_bytes, AES.MODE_CBC, IV=iv)
        ciphertext = encryptor.encrypt(text_bytes)
        return ciphertext.hex().upper()

    @staticmethod
    def xor_str(s: bytes, t: bytes) -> bytes:
        return bytes([_a ^ _b for _a, _b in zip(s, t)])

    @staticmethod
    def calc_opc_hex(k_hex: str, op_hex: str) -> str:
        iv = binascii.unhexlify(16 * "00")
        ki = binascii.unhexlify(k_hex)
        op = binascii.unhexlify(op_hex)
        aes_crypt = AES.new(ki, mode=AES.MODE_CBC, IV=iv)
        data = op
        o_pc = CryptoUtils.xor_str(data, aes_crypt.encrypt(data))
        return o_pc.hex().upper()


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


__all__ = ["DependentDataGenerator"]
