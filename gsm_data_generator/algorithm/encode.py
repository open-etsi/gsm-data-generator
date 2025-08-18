from typing import Optional

from ..transform import DataTransform


class EncodingUtils:
    @staticmethod
    def enc_pin(pin: str) -> str:
        return DataTransform.rpad(DataTransform.s2h(pin), 16).upper()

    @staticmethod
    def dec_pin(encoded_pin: str) -> str:
        return DataTransform.h2s(encoded_pin).upper()

    @staticmethod
    def enc_imsi(imsi):
        imsi = str(imsi)
        l = DataTransform.half_round_up(
            len(imsi) + 1
        )  # Required bytes - include space for odd/even indicator
        oe = len(imsi) & 1  # Odd (1) / Even (0)
        ei = "%02x" % l + DataTransform.swap_nibbles(
            "%01x%s" % ((oe << 3) | 1, DataTransform.rpad(imsi, 15))
        )
        return ei

    @staticmethod
    def dec_imsi(ef: str) -> Optional[str]:
        if len(ef) < 4:
            return None
        l = int(ef[0:2], 16) * 2 - 1
        swapped = DataTransform.swap_nibbles(ef[2:]).rstrip("f")
        if len(swapped) < 1:
            return None
        oe = (int(swapped[0]) >> 3) & 1
        if not oe:
            l -= 1
        if l != len(swapped) - 1:
            return None
        return swapped[1:]

    @staticmethod
    def enc_iccid(iccid: str) -> str:
        luhn = EncodingUtils.calculate_luhn(iccid)
        iccid_with_luhn = iccid + str(luhn)
        return DataTransform.swap_nibbles(
            DataTransform.rpad(iccid_with_luhn, 20)
        ).upper()

    @staticmethod
    def dec_iccid(ef: str) -> str:
        return DataTransform.swap_nibbles(ef.upper()).strip("F")[:-1]

    @staticmethod
    def calculate_luhn(cc: str) -> int:
        num = list(map(int, str(cc)))
        check_digit = (
            10 - sum(num[-2::-2] + [sum(divmod(d * 2, 10)) for d in num[::-2]]) % 10
        )
        return 0 if check_digit == 10 else check_digit


__all__ = ["EncodingUtils"]
