
from typing import Tuple

class LibraryInfo:
    # __version__ = "0.0.1.dev0"

    _supported_types: Tuple[str, ...] = (
        "IMSI",
        "ICCID",
        "PIN1",
        "PUK1",
        "PIN2",
        "PUK2",
        "ADM1",
        "ADM6",
        "KI",
        "OPC",
        "ACC",
        "KIC1",
        "KID1",
        "KIK1",
        "KIC2",
        "KID2",
        "KIK2",
        "KIC3",
        "KID3",
        "KIK3",
    )

    @classmethod
    def get_supported_types(cls) -> Tuple[str, ...]:
        """Return all supported GSM data types."""
        return cls._supported_types
