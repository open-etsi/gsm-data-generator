import secrets


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


__all__ = ["DataGenerator"]
