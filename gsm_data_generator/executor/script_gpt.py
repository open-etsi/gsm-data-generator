import pandas as pd
import logging
from typing import Dict, Tuple

from ..generator import (
    CryptoUtils,
    DataGenerator,
    DataProcessing,
    DataFrameProcessor,
    DependentDataGenerator,
)
from ..globals import DataFrames, Parameters
from ..utils import DEFAULT_HEADER, copy_function, list_2_dict

logger = logging.getLogger(__name__)


class DataGenerationScript:
    """
    Main orchestration class for GSM data generation.

    Responsibilities:
    - Map configuration into global Parameters
    - Generate demo or non-demo data
    - Apply cryptographic and random generators
    - Post-process DataFrames into final datasets
    """

    def __init__(self, config_holder, params=None, dataframes=None):
        self.config_holder = config_holder
        self.params = params or Parameters.get_instance()
        self.dataframes = dataframes or DataFrames.get_instance()
        self.crypto_utils = CryptoUtils()
        self.data_generator = DataGenerator()
        self.data_processor = DataProcessing()
        self.df_processor = DataFrameProcessor()
        self.dep_data_generator = DependentDataGenerator()

    # ---------------------------
    # CONFIG → PARAMETERS
    # ---------------------------
    def json_to_global_params(self) -> None:
        """Map DISP + PARAMETERS config into global Parameters singleton."""
        disp = self.config_holder.DISP
        prm = self.params

        prm.set_SERVER_SEP(disp.server_data_sep)
        prm.set_ELECT_SEP(disp.elect_data_sep)
        prm.set_GRAPH_SEP(disp.graph_data_sep)

        prm.set_K4(disp.K4)
        prm.set_OP(disp.op)
        prm.set_IMSI(disp.imsi)
        prm.set_ICCID(disp.iccid)
        prm.set_PIN1(disp.pin1)
        prm.set_PUK1(disp.puk1)
        prm.set_PIN2(disp.pin2)
        prm.set_PUK2(disp.puk2)
        prm.set_ADM1(disp.adm1)
        prm.set_ADM6(disp.adm6)
        prm.set_DATA_SIZE(disp.size)

        # Production flag
        prm.set_PRODUCTION_CHECK(False)

        # Checks
        prm.set_ELECT_CHECK(disp.elect_check)
        prm.set_GRAPH_CHECK(disp.graph_check)
        prm.set_SERVER_CHECK(disp.server_check)

        # Dictionaries
        prm.set_ELECT_DICT(list_2_dict(self.config_holder.PARAMETERS.data_variables))
        prm.set_GRAPH_DICT(self.config_holder.PARAMETERS.laser_variables)
        prm.set_SERVER_DICT(list_2_dict(self.config_holder.PARAMETERS.server_variables))

        # Fix flags
        prm.set_PIN1_RAND(disp.pin1_fix)
        prm.set_PUK1_RAND(disp.puk1_fix)
        prm.set_PIN2_RAND(disp.pin2_fix)
        prm.set_PUK2_RAND(disp.puk2_fix)
        prm.set_ADM1_RAND(disp.adm1_fix)
        prm.set_ADM6_RAND(disp.adm6_fix)

    # ---------------------------
    # GENERATORS
    # ---------------------------
    def generate_code(self, code_type: str, length: int) -> str:
        """
        Generic generator for PIN/PUK/ADM codes.
        Uses fixed value from Parameters if *_RAND flag is set.
        """
        if getattr(self.params, f"get_{code_type}_RAND")():
            return getattr(self.params, f"get_{code_type}")()

        generator = {
            4: self.data_generator.generate_4_digit,
            8: self.data_generator.generate_8_digit,
        }[length]
        return generator()

    def generate_eki(self, ki: str) -> str:
        return self.dep_data_generator.calculate_eki(self.params.get_K4(), ki)

    def generate_opc(self, ki: str) -> str:
        return self.dep_data_generator.calculate_opc(self.params.get_OP(), ki)

    # ---------------------------
    # DATAFRAME PROCESSING
    # ---------------------------
    def apply_function(
        self, df: pd.DataFrame, dest: str, src: str, function
    ) -> pd.DataFrame:
        """Apply transformation function on `src` column to produce `dest` column."""
        if dest in df.columns:
            df[dest] = df[src].apply(function)
        return df

    def apply_functions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all necessary transformations to a dataframe."""
        df["ICCID"] = df["ICCID"].apply(copy_function)
        df["IMSI"] = df["IMSI"].apply(copy_function)

        df["PIN1"] = df["PIN1"].apply(lambda _: self.generate_code("PIN1", 4))
        df["PIN2"] = df["PIN2"].apply(lambda _: self.generate_code("PIN2", 4))
        df["PUK1"] = df["PUK1"].apply(lambda _: self.generate_code("PUK1", 8))
        df["PUK2"] = df["PUK2"].apply(lambda _: self.generate_code("PUK2", 8))
        df["ADM1"] = df["ADM1"].apply(lambda _: self.generate_code("ADM1", 8))
        df["ADM6"] = df["ADM6"].apply(lambda _: self.generate_code("ADM6", 8))

        df["KI"] = df["KI"].apply(lambda _: self.data_generator.generate_ki())
        df["ACC"] = df["IMSI"].apply(
            lambda imsi: self.dep_data_generator.calculate_acc(imsi=str(imsi))
        )

        # Apply EKI / OPC
        self.apply_function(df, "EKI", "KI", self.generate_eki)
        self.apply_function(df, "OPC", "KI", self.generate_opc)

        # OTA keys
        for i in range(1, 4):
            for key in ["KIC", "KID", "KIK"]:
                col = f"{key}{i}"
                if col in df.columns:
                    df[col] = df["KI"].apply(
                        lambda _: self.data_generator.generate_otas()
                    )

        return df
