# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel, field_validator, constr, Field
from threading import Lock

debug = False


class DataFrames:
    __instance = None

    def __init__(self):
        if DataFrames.__instance is not None:
            raise Exception(
                "GlobalParameters class is a singleton! Use get_instance() to access the instance."
            )
        else:
            DataFrames.__instance = self
            self.__INPUT_DF = pd.DataFrame()
            self.ELECT_DF = pd.DataFrame()
            self.GRAPH_DF = pd.DataFrame()
            self.SERVER_DF = pd.DataFrame()
            self.__KEYS = {}

    @staticmethod
    def get_instance():
        if DataFrames.__instance is None:
            DataFrames.__instance = DataFrames()
        return DataFrames.__instance

    def set_ELECT_DF(self, value):
        self.ELECT_DF = pd.DataFrame()
        self.ELECT_DF = value

    def get_ELECT_DF(self):
        return self.ELECT_DF

    def set_GRAPH_DF(self, value):
        self.GRAPH_DF = pd.DataFrame()
        self.GRAPH_DF = value

    def get_GRAPH_DF(self):
        return self.GRAPH_DF

    def set_SERVER_DF(self, value):
        self.SERVER_DF = pd.DataFrame()
        self.SERVER_DF = value

    def get_SERVER_DF(self):
        return self.SERVER_DF

    def set_KEYS(self, value):
        self.__KEYS = value

    def get_KEYS(self):
        return self.__KEYS

    def set_input_df(self, value):
        self.__INPUT_DF = pd.DataFrame()
        self.__INPUT_DF = value

    def get_input_df(self):
        return self.__INPUT_DF


class Parameters(DataFrames):
    """
    Singleton class to hold global SIM/USIM configuration parameters.
    Provides thread-safe access to mutable configuration attributes.
    """

    __instance = None
    __lock = Lock()

    def __init__(self):
        super().__init__()
        self.__def_head = None
        if Parameters.__instance is not None:
            raise Exception(
                "GlobalParameters class is a singleton! Use get_instance() to access the instance."
            )
        else:
            Parameters.__instance = self

            self.__ICCID: constr(strip_whitespace=True, min_length=18, max_length=20) = ""  # type: ignore
            self.__IMSI: constr(strip_whitespace=True, min_length=15, max_length=15) = ""  # type: ignore
            self.__PIN1: constr(regex=r"^\d{4}$") = ""  # type: ignore
            self.__PUK1: constr(regex=r"^\d{8}$") = ""  # type: ignore
            self.__PIN2: constr(regex=r"^\d{4}$") = ""  # type: ignore
            self.__PUK2: constr(regex=r"^\d{8}$") = ""  # type: ignore
            self.__K4: str = ""
            self.__OP: str = ""
            self.__ADM1: str = ""
            self.__ADM6: str = ""
            self.__ACC: str = ""
            self.__DATA_SIZE: str = ""

            self.__ELECT_CHECK: bool = False
            self.__GRAPH_CHECK: bool = False
            self.__SERVER_CHECK: bool = False

            self.__pin1_rand: bool = True
            self.__puk1_rand: bool = True
            self.__pin2_rand: bool = True
            self.__puk2_rand: bool = True
            self.__adm1_rand: bool = True
            self.__adm6_rand: bool = True
            self.__acc_rand: bool = True

            self.__ELECT_DICT: dict = {}
            self.__GRAPH_DICT: dict = {}
            self.__SERVER_DICT: dict = {}

            self.file_name: str

            self.__ELECT_SEP: str = ""
            self.__GRAPH_SEP: str = ""
            self.__SERVR_SEP: str = ""

    @field_validator("DATA_SIZE")
    def check_data_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("DATA_SIZE must be a positive integer")
        return v

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.__ICCID and not (18 <= len(self.__ICCID) <= 20):
            raise ValueError("ICCID must be 18–20 characters long")
        if self.__IMSI and len(self.__IMSI) != 15:
            raise ValueError("IMSI must be exactly 15 digits")

    @classmethod
    def get_instance(cls):
        """
        Thread-safe access to the singleton instance.
        """
        with cls.__lock:
            if cls.__instance is None:
                cls.__instance = Parameters()
            return cls.__instance

    # @staticmethod
    # def _to_str(value: str) -> str:
    #     if not isinstance(value, str):
    #         raise TypeError(f"Expected str, got {type(value).__name__}")
    #     return value.strip()

    # @staticmethod
    # def _to_bool(value: bool) -> bool:
    #     if not isinstance(value, bool):
    #         raise TypeError(f"Expected bool, got {type(value).__name__}")
    #     return value

    # @staticmethod
    # def _to_dict(value: dict) -> dict:
    #     if not isinstance(value, dict):
    #         raise TypeError(f"Expected dict, got {type(value).__name__}")
    #     return value

    @property
    def ELECT_SEP(self) -> str:
        """Separator string for ELECT file generation."""
        return self.__ELECT_SEP

    @ELECT_SEP.setter
    def ELECT_SEP(self, value: str) -> None:
        self.__ELECT_SEP = value

    @property
    def GRAPH_SEP(self) -> str:
        """Separator string for GRAPH file generation."""
        return self.__GRAPH_SEP

    @GRAPH_SEP.setter
    def GRAPH_SEP(self, value: str) -> None:
        self.__GRAPH_SEP = value

    @property
    def SERVER_SEP(self) -> str:
        """Separator string for SERVER file generation."""
        return self.__SERVER_SEP

    @SERVER_SEP.setter
    def SERVER_SEP(self, value: str) -> None:
        self.__SERVER_SEP = value

    @property
    def TEMPLATE_JSON(self) -> str:
        """Path to the template JSON file."""
        return self.__TEMPLATE_JSON

    @TEMPLATE_JSON.setter
    def TEMPLATE_JSON(self, value: str) -> None:
        self.__TEMPLATE_JSON = value

    @property
    def INPUT_FILE_PATH(self) -> str:
        """Path to the input file."""
        return self.__INPUT_FILE_PATH

    @INPUT_FILE_PATH.setter
    def INPUT_FILE_PATH(self, value: str) -> None:
        self.__INPUT_FILE_PATH = value

    @property
    def OUTPUT_FILES_DIR(self) -> str:
        """Directory for generated output files."""
        return self.__OUTPUT_FILES_DIR

    @OUTPUT_FILES_DIR.setter
    def OUTPUT_FILES_DIR(self, value: str) -> None:
        self.__OUTPUT_FILES_DIR = value

    @property
    def ICCID(self) -> str:
        """ICCID value (Integrated Circuit Card Identifier)."""
        return self.__ICCID

    @ICCID.setter
    def ICCID(self, value: str) -> None:
        self.__ICCID = value

    @property
    def IMSI(self) -> str:
        """IMSI value (International Mobile Subscriber Identity)."""
        return self.__IMSI

    @IMSI.setter
    def IMSI(self, value: str) -> None:
        self.__IMSI = value

    @property
    def PIN1(self) -> str:
        """PIN1 value."""
        return self.__PIN1

    @PIN1.setter
    def PIN1(self, value: str) -> None:
        self.__PIN1 = value

    @property
    def PUK1(self) -> str:
        """PUK1 value."""
        return self.__PUK1

    @PUK1.setter
    def PUK1(self, value: str) -> None:
        self.__PUK1 = value

    @property
    def PIN2(self) -> str:
        """PIN2 value."""
        return self.__PIN2

    @PIN2.setter
    def PIN2(self, value: str) -> None:
        self.__PIN2 = value

    @property
    def PUK2(self) -> str:
        """PUK2 value."""
        return self.__PUK2

    @PUK2.setter
    def PUK2(self, value: str) -> None:
        self.__PUK2 = value

    @property
    def OP(self) -> str:
        """Operator Code (OP)."""
        return self.__OP

    @OP.setter
    def OP(self, value: str) -> None:
        self.__OP = value

    @property
    def K4(self) -> str:
        """K4 Transport Key."""
        return self.__K4

    @K4.setter
    def K4(self, value: str) -> None:
        self.__K4 = value

    @property
    def ELECT_CHECK(self) -> bool:
        """Flag to enable or disable ELECT check."""
        return self.__ELECT_CHECK

    @ELECT_CHECK.setter
    def ELECT_CHECK(self, value: bool) -> None:
        self.__ELECT_CHECK = value

    @property
    def GRAPH_CHECK(self) -> bool:
        """Flag to enable or disable GRAPH check."""
        return self.__GRAPH_CHECK

    @GRAPH_CHECK.setter
    def GRAPH_CHECK(self, value: bool) -> None:
        self.__GRAPH_CHECK = value

    @property
    def SERVER_CHECK(self) -> bool:
        """Flag to enable or disable SERVER check."""
        return self.__SERVER_CHECK

    @SERVER_CHECK.setter
    def SERVER_CHECK(self, value: bool) -> None:
        self.__SERVER_CHECK = value

    @property
    def PIN1_RAND(self) -> bool:
        """Enable/disable randomization of PIN1."""
        return self.__pin1_rand

    @PIN1_RAND.setter
    def PIN1_RAND(self, value: bool) -> None:
        self.__pin1_rand = value

    @property
    def ELECT_DICT(self) -> dict:
        """Dictionary containing ELECT data."""
        return self.__ELECT_DICT

    @ELECT_DICT.setter
    def ELECT_DICT(self, value: dict) -> None:
        self.__ELECT_DICT = value

    @property
    def PUK1_RAND(self) -> bool:
        """Enable/disable randomization of PUK1."""
        return self.__puk1_rand

    @PUK1_RAND.setter
    def PUK1_RAND(self, value: bool) -> None:
        self.__puk1_rand = value

    @property
    def PIN2_RAND(self) -> bool:
        """Enable/disable randomization of PIN2."""
        return self.__pin2_rand

    @PIN2_RAND.setter
    def PIN2_RAND(self, value: bool) -> None:
        self.__pin2_rand = value

    @property
    def PUK2_RAND(self) -> bool:
        """Enable/disable randomization of PUK2."""
        return self.__puk2_rand

    @PUK2_RAND.setter
    def PUK2_RAND(self, value: bool) -> None:
        self.__puk2_rand = value

    @property
    def ADM1_RAND(self) -> bool:
        """Enable/disable randomization of ADM1."""
        return self.__adm1_rand

    @ADM1_RAND.setter
    def ADM1_RAND(self, value: bool) -> None:
        self.__adm1_rand = value

    @property
    def ADM6_RAND(self) -> bool:
        """Enable/disable randomization of ADM6."""
        return self.__adm6_rand

    @ADM6_RAND.setter
    def ADM6_RAND(self, value: bool) -> None:
        self.__adm6_rand = value

    @property
    def ACC_RAND(self) -> bool:
        """Enable/disable randomization of ACC."""
        return self.__acc_rand

    @ACC_RAND.setter
    def ACC_RAND(self, value: bool) -> None:
        self.__acc_rand = value

    @property
    def GRAPH_DICT(self) -> dict:
        """Graph configuration dictionary."""
        return self.__GRAPH_DICT

    @GRAPH_DICT.setter
    def GRAPH_DICT(self, value: dict) -> None:
        self.__GRAPH_DICT = value

    @property
    def SERVER_DICT(self) -> dict:
        """Server configuration dictionary."""
        return self.__SERVER_DICT

    @SERVER_DICT.setter
    def SERVER_DICT(self, value: dict) -> None:
        self.__SERVER_DICT = value

    @property
    def SERVR_SEP(self) -> str:
        """Separator string for server file generation."""
        return self.__SERVR_SEP

    @SERVR_SEP.setter
    def SERVR_SEP(self, value: str) -> None:
        self.__SERVR_SEP = value

    @property
    def ADM1(self) -> str:
        """ADM1 code string."""
        return self.__ADM1

    @ADM1.setter
    def ADM1(self, value: str) -> None:
        self.__ADM1 = value

    @property
    def ADM6(self) -> str:
        """ADM6 code string."""
        return self.__ADM6

    @ADM6.setter
    def ADM6(self, value: str) -> None:
        self.__ADM6 = value

    @property
    def ACC(self) -> str:
        """ACC code string."""
        return self.__ACC

    @ACC.setter
    def ACC(self, value: str) -> None:
        self.__ACC = value

    @property
    def DATA_SIZE(self) -> str:
        """Size of the data block or record."""
        return self.__DATA_SIZE

    @DATA_SIZE.setter
    def DATA_SIZE(self, value: str) -> None:
        self.__DATA_SIZE = value

    def get_all_params_dict(self) -> dict:
        param_dict = {
            #            "Demo Data": self.get_PRODUCTION_CHECK(),
            "Demo Data": True,
            "OP": self.OP,
            "K4": self.K4,
            "ICCID": self.ICCID,
            "IMSI": self.IMSI,
            "PIN1": self.PIN1,
            "PUK1": self.PUK1,
            "PIN2": self.PIN2,
            "PUK2": self.PUK2,
            "ADM1": self.ADM1,
            "ADM6": self.ADM6,
            "ACC": self.ACC,
            "DATA_SIZE": self.DATA_SIZE,
            #            "INPUT_PATH": self.INPUT_PATH,
        }
        print(param_dict)
        return param_dict

    @staticmethod
    def is_valid(param1, param_name: str) -> bool:
        result = False
        param = param1
        match param_name:
            case "ICCID":
                result = (
                    len(str(param)) == 20
                    or len(str(param)) == 19
                    or len(str(param)) == 18
                )
            case "IMSI":
                result = len(str(param)) == 15
            case "PIN1" | "PIN2":
                result = len(str(param)) == 4
            case "PUK1" | "PUK2" | "ADM1" | "ADM6":
                result = len(str(param)) == 8
            case "OP":
                result = len(str(param)) == 32
            case "K4":
                result = (len(str(param)) == 64) or (len(str(param)) == 32)
            case "SIZE":
                param = int(param)
                result = len(str(param)) != 0 or param > 0
            case "DICT":
                param = dict(param)
                result = len(param) > 0
        print(param_name, result)
        return result

    @staticmethod
    def is_valid_df(param, param_name: str) -> bool:
        match param_name:
            case "DF":
                df = pd.DataFrame(param)
                return df.empty
            case _:
                # Default behavior if param_name doesn't match any known case
                return False

    def print_all_global_parameters(self) -> None:
        """
        Print all major configuration parameters for debugging.
        """
        print("======= Current Global Parameters =======")
        params = {
            "IMSI": self.IMSI,
            "ICCID": self.ICCID,
            "PIN1": self.PIN1,
            "PUK1": self.PUK1,
            "PIN2": self.PIN2,
            "PUK2": self.PUK2,
            "ADM1": self.ADM1,
            "ADM6": self.ADM6,
            "OP": self.OP,
            "K4": self.K4,
            "DATA_SIZE": self.DATA_SIZE,
            "ELECT_DICT": self.ELECT_DICT,
            "GRAPH_DICT": self.GRAPH_DICT,
            "SERVER_DICT": self.SERVER_DICT,
        }
        for key, value in params.items():
            print(f"{key}: {value}")

    def check_params(self) -> bool:
        result = (
            self.is_valid(self.IMSI, "IMSI")
            and self.is_valid(self.ICCID, "ICCID")
            and self.is_valid(self.DATA_SIZE, "SIZE")
            and self.is_valid(self.PIN1, "PIN1")
            and self.is_valid(self.PUK1, "PUK1")
            and self.is_valid(self.PIN2, "PIN2")
            and self.is_valid(self.PUK2, "PUK2")
            and self.is_valid(self.ADM1, "ADM1")
            and self.is_valid(self.ADM6, "ADM6")
            and self.is_valid(self.OP, "OP")
            and self.is_valid(self.K4, "K4")
            and self.is_valid(self.ELECT_DICT, "DICT")
            and self.is_valid(self.GRAPH_DICT, "DICT")
        )
        return result


__all__ = ["Parameters", "DataFrames"]
