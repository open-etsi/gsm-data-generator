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

    # def is_VALID_DF(self, param, param_name: str) -> bool:
    #     if param_name == "DF":
    #         return param.empty
    #     else:
    #         return False


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
            # self.__PROD_CHECK: bool = False

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

    # Example validator
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
        # if self.__DATA_SIZE < 0:
        #     raise ValueError("DATA_SIZE must be a positive integer")

    @classmethod
    def get_instance(cls):
        """
        Thread-safe access to the singleton instance.
        """
        with cls.__lock:
            if cls.__instance is None:
                cls.__instance = Parameters()
            return cls.__instance

    def set_ELECT_SEP(self, value: str) -> None:
        """
        Set the separator string for ELECT file generation.

        Args:
            value (str): Separator string to be used.
        """
        self.__ELECT_SEP = str(value)

    def get_ELECT_SEP(self) -> str:
        """
        Get the separator string for ELECT file generation.

        Returns:
            str: Current ELECT separator string.
        """
        return self.__ELECT_SEP

    def set_GRAPH_SEP(self, value: str) -> None:
        """
        Set the separator string for GRAPH file generation.

        Args:
            value (str): Separator string to be used.
        """
        self.__GRAPH_SEP = str(value)

    def get_GRAPH_SEP(self) -> str:
        """
        Get the separator string for GRAPH file generation.

        Returns:
            str: Current GRAPH separator string.
        """
        return self.__GRAPH_SEP

    def set_SERVER_SEP(self, value: str) -> None:
        """
        Set the separator string for SERVER file generation.

        Args:
            value (str): Separator string to be used.
        """
        self.__SERVR_SEP = str(value)

    def get_SERVER_SEP(self) -> str:
        """
        Get the separator string for SERVER file generation.

        Returns:
            str: Current SERVER separator string.
        """
        return self.__SERVR_SEP

    def set_TEMPLATE_JSON(self, value: str) -> None:
        """
        Set the path to the template JSON file.

        Args:
            value (str): Path to template JSON file.
        """
        self.__TEMPLATE_JSON = str(value)

    def get_TEMPLATE_JSON(self) -> str:
        """
        Get the path to the template JSON file.

        Returns:
            str: Path of template JSON file.
        """
        return self.__TEMPLATE_JSON

    def set_INPUT_FILE_PATH(self, value: str) -> None:
        """
        Set the input file path.

        Args:
            value (str): Path to the input file.
        """
        self.__INPUT_FILE_PATH = str(value)

    def get_INPUT_FILE_PATH(self) -> str:
        """
        Get the input file path.

        Returns:
            str: Path of input file.
        """
        return self.__INPUT_FILE_PATH

    def set_OUTPUT_FILES_DIR(self, value: str) -> None:
        """
        Set the output directory for generated files.

        Args:
            value (str): Directory path.
        """
        self.__OUTPUT_FILES_DIR = str(value)

    def get_OUTPUT_FILES_DIR(self) -> str:
        """
        Get the output directory for generated files.

        Returns:
            str: Path of output files directory.
        """
        return self.__OUTPUT_FILES_DIR

    # ------------------ SIM PARAMETERS ------------------

    def set_ICCID(self, value: str) -> None:
        """Set the ICCID value."""
        self.__ICCID = str(value)

    def get_ICCID(self) -> str:
        """Get the ICCID value."""
        return self.__ICCID

    def set_IMSI(self, value: str) -> None:
        """Set the IMSI value."""
        self.__IMSI = str(value)

    def get_IMSI(self) -> str:
        """Get the IMSI value."""
        return self.__IMSI

    def set_PIN1(self, value: str) -> None:
        """Set the PIN1 value."""
        self.__PIN1 = str(value)

    def get_PIN1(self) -> str:
        """Get the PIN1 value."""
        return self.__PIN1

    def set_PUK1(self, value: str) -> None:
        """Set the PUK1 value."""
        self.__PUK1 = str(value)

    def get_PUK1(self) -> str:
        """Get the PUK1 value."""
        return self.__PUK1

    def set_PIN2(self, value: str) -> None:
        """Set the PIN2 value."""
        self.__PIN2 = str(value)

    def get_PIN2(self) -> str:
        """Get the PIN2 value."""
        return self.__PIN2

    def set_PUK2(self, value: str) -> None:
        """Set the PUK2 value."""
        self.__PUK2 = str(value)

    def get_PUK2(self) -> str:
        """Get the PUK2 value."""
        return self.__PUK2

    def set_OP(self, value: str) -> None:
        """Set the Operator Code (OP)."""
        self.__OP = str(value)

    def get_OP(self) -> str:
        """Get the Operator Code (OP)."""
        return self.__OP

    def set_K4(self, value: str) -> None:
        """Set the K4 (Transport Key)."""
        self.__K4 = str(value)

    def get_K4(self) -> str:
        """Get the K4 (Transport Key)."""
        return self.__K4

    def set_ADM1(self, value: str) -> None:
        """Set the ADM1 value."""
        self.__ADM1 = str(value)

    def get_ADM1(self) -> str:
        """Get the ADM1 value."""
        return self.__ADM1

    def set_ADM6(self, value: str) -> None:
        """Set the ADM6 value."""
        self.__ADM6 = str(value)

    def get_ADM6(self) -> str:
        """Get the ADM6 value."""
        return self.__ADM6

    def set_ACC(self, value: str) -> None:
        """Set the ACC (Access Control Class)."""
        self.__ACC = str(value)

    def get_ACC(self) -> str:
        """Get the ACC (Access Control Class)."""
        return self.__ACC

    # ------------------ FLAGS ------------------

    def set_ELECT_CHECK(self, value: bool) -> None:
        """Enable or disable ELECT check flag."""
        self.__ELECT_CHECK = value

    def get_ELECT_CHECK(self) -> bool:
        """Get the ELECT check flag status."""
        return self.__ELECT_CHECK

    def set_GRAPH_CHECK(self, value: bool) -> None:
        """Enable or disable GRAPH check flag."""
        self.__GRAPH_CHECK = value

    def get_GRAPH_CHECK(self) -> bool:
        """Get the GRAPH check flag status."""
        return self.__GRAPH_CHECK

    def set_SERVER_CHECK(self, value: bool) -> None:
        """Enable or disable SERVER check flag."""
        self.__SERVER_CHECK = value

    def get_SERVER_CHECK(self) -> bool:
        """Get the SERVER check flag status."""
        return self.__SERVER_CHECK

    # ------------------ OTHER ------------------

    def set_DATA_SIZE(self, value: str) -> None:
        """
        Set the data size configuration.

        Args:
            value (int): Data size value.
        """
        self.__DATA_SIZE = value

    def get_DATA_SIZE(self) -> str:
        """
        Get the data size configuration.

        Returns:
            int: Current data size value.
        """
        return self.__DATA_SIZE

    # def set_DEFAULT_HEADER(self, value: list) -> None:
    #     """
    #     Set the default header configuration.

    #     Args:
    #         value (list): List of header values.
    #     """
    #     self.__def_head = value

    # def get_DEFAULT_HEADER(self) -> list:
    #     """
    #     Get the default header configuration.

    #     Returns:
    #         list: List of header values.
    #     """
    #     return self.__def_head

    def set_PIN1_RAND(self, value: bool):
        """
        Enable or disable randomization of PIN1.

        Args:
            value (bool):
                - True → PIN1 will be randomly generated.
                - False → PIN1 will use a fixed/manual value.
        """
        self.__pin1_rand = value

    def get_PIN1_RAND(self) -> bool:
        """
        Get the current randomization status of PIN1.

        Returns:
            bool:
                - True if PIN1 randomization is enabled.
                - False if PIN1 is fixed/manual.
        """
        return self.__pin1_rand

    def set_PUK1_RAND(self, value: bool):
        """
        Enable or disable randomization of PUK1.

        Args:
            value (bool):
                - True → PUK1 will be randomly generated.
                - False → PUK1 will use a fixed/manual value.
        """
        self.__puk1_rand = value

    def get_PUK1_RAND(self) -> bool:
        """
        Get the current randomization status of PUK1.

        Returns:
            bool:
                - True if PUK1 randomization is enabled.
                - False if PUK1 is fixed/manual.
        """
        return self.__puk1_rand

    def set_PIN2_RAND(self, value: bool):
        """
        Enable or disable randomization of PIN2.

        Args:
            value (bool):
                - True → PIN2 will be randomly generated.
                - False → PIN2 will use a fixed/manual value.
        """
        self.__pin2_rand = value

    def get_PIN2_RAND(self) -> bool:
        """
        Get the current randomization status of PIN2.

        Returns:
            bool:
                - True if PIN2 randomization is enabled.
                - False if PIN2 is fixed/manual.
        """
        return self.__pin2_rand

    def set_PUK2_RAND(self, value: bool):
        """
        Enable or disable randomization of PUK2.

        Args:
            value (bool):
                - True → PUK2 will be randomly generated.
                - False → PUK2 will use a fixed/manual value.
        """
        self.__puk2_rand = value

    def get_PUK2_RAND(self) -> bool:
        """
        Get the current randomization status of PUK2.

        Returns:
            bool:
                - True if PUK2 randomization is enabled.
                - False if PUK2 is fixed/manual.
        """
        return self.__puk2_rand

    def set_ADM1_RAND(self, value: bool):
        """
        Enable or disable randomization of ADM1.

        Args:
            value (bool):
                - True → ADM1 will be randomly generated.
                - False → ADM1 will use a fixed/manual value.
        """
        self.__adm1_rand = value

    def get_ADM1_RAND(self) -> bool:
        """
        Get the current randomization status of ADM1.

        Returns:
            bool:
                - True if ADM1 randomization is enabled.
                - False if ADM1 is fixed/manual.
        """
        return self.__adm1_rand

    def set_ADM6_RAND(self, value: bool):
        """
        Enable or disable randomization of ADM6.

        Args:
            value (bool):
                - True → ADM6 will be randomly generated.
                - False → ADM6 will use a fixed/manual value.
        """
        self.__adm6_rand = value

    def get_ADM6_RAND(self) -> bool:
        """
        Get the current randomization status of ADM6.

        Returns:
            bool:
                - True if ADM6 randomization is enabled.
                - False if ADM6 is fixed/manual.
        """
        return self.__adm6_rand

    def set_ACC_RAND(self, value: bool):
        """
        Enable or disable randomization of ACC.

        Args:
            value (bool):
                - True → ACC will be randomly generated.
                - False → ACC will use a fixed/manual value.
        """
        self.__acc_rand = value

    def get_ACC_RAND(self) -> bool:
        """
        Get the current randomization status of ACC.

        Returns:
            bool:
                - True if ACC randomization is enabled.
                - False if ACC is fixed/manual.
        """
        return self.__acc_rand

    def set_INPUT_PATH(self, value: str) -> None:
        """
        Set the input path.

        Args:
            value (str): Path to the input resource.
        """
        self.__INPUT_PATH = value

    def get_INPUT_PATH(self) -> str:
        """
        Get the input path.

        Returns:
            str: Current input path.
        """
        return self.__INPUT_PATH


    def set_LASER_EXT_PATH(self, value: str) -> None:
        """
        Set the laser extraction path.

        Args:
            value (str): Path for laser extraction data.
        """
        self.__LASER_EXT_PATH = value

    def get_LASER_EXT_PATH(self) -> str:
        """
        Get the laser extraction path.

        Returns:
            str: Current laser extraction path.
        """
        return self.__LASER_EXT_PATH


    def set_ELECT_DICT(self, value: dict) -> None:
        """
        Set the ELECT dictionary configuration.

        Args:
            value (dict): Dictionary containing ELECT data.
        """
        self.__ELECT_DICT = value

    def get_ELECT_DICT(self) -> dict:
        """
        Get the ELECT dictionary configuration.

        Returns:
            dict: ELECT data dictionary.
        """
        return self.__ELECT_DICT


    def set_GRAPH_DICT(self, value: dict) -> None:
        """
        Set the GRAPH dictionary configuration.

        Args:
            value (dict): Dictionary containing GRAPH data.
        """
        self.__GRAPH_DICT = value

    def get_GRAPH_DICT(self) -> dict:
        """
        Get the GRAPH dictionary configuration.

        Returns:
            dict: GRAPH data dictionary.
        """
        return self.__GRAPH_DICT


    def set_SERVER_DICT(self, value: dict) -> None:
        """
        Set the SERVER dictionary configuration.

        Args:
            value (dict): Dictionary containing SERVER data.
        """
        self.__SERVER_DICT = value

    def get_SERVER_DICT(self) -> dict:
        """
        Get the SERVER dictionary configuration.

        Returns:
            dict: SERVER data dictionary.
        """
        return self.__SERVER_DICT


    def set_file_name(self, value: str) -> None:
        """
        Set the file name.

        Args:
            value (str): Name of the file.
        """
        self.file_name = value

    def get_file_name(self) -> str:
        """
        Get the file name.

        Returns:
            str: Current file name.
        """
        return self.file_name

    def get_all_params_dict(self) -> dict:
        param_dict = {
            #            "Demo Data": self.get_PRODUCTION_CHECK(),
            "Demo Data": True,
            "OP": self.get_OP(),
            "K4": self.get_K4(),
            "ICCID": self.get_ICCID(),
            "IMSI": self.get_IMSI(),
            "PIN1": self.get_PIN1(),
            "PUK1": self.get_PUK1(),
            "PIN2": self.get_PIN2(),
            "PUK2": self.get_PUK2(),
            "ADM1": self.get_ADM1(),
            "ADM6": self.get_ADM6(),
            "ACC": self.get_ACC(),
            "DATA_SIZE": self.get_DATA_SIZE(),
            "INPUT_PATH": self.get_INPUT_PATH(),
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

    def check_params(self) -> bool:
        #        if not self.get_PRODUCTION_CHECK():
        if False:
            print("===============Production===============")
            result = (
                #                self.is_valid(self.get_IMSI(), "IMSI")
                #                and self.is_valid(self.get_ICCID(), "ICCID")
                self.is_valid(self.get_PIN1(), "PIN1")
                and self.is_valid(self.get_PUK1(), "PUK1")
                and self.is_valid(self.get_PIN2(), "PIN2")
                and self.is_valid(self.get_PUK2(), "PUK2")
                and self.is_valid(self.get_ADM1(), "ADM1")
                and self.is_valid(self.get_ADM6(), "ADM6")
                and self.is_valid(self.get_OP(), "OP")
                and self.is_valid(self.get_K4(), "K4")
                #                and self.is_valid(self.get_DATA_SIZE(), "SIZE")
                and self.is_valid(self.get_ELECT_DICT(), "DICT")
                and self.is_valid(self.get_GRAPH_DICT(), "DICT")
            )
        else:
            print("=================Demo===================")
            result = (
                self.is_valid(self.get_IMSI(), "IMSI")
                and self.is_valid(self.get_ICCID(), "ICCID")
                and self.is_valid(self.get_DATA_SIZE(), "SIZE")
                and self.is_valid(self.get_PIN1(), "PIN1")
                and self.is_valid(self.get_PUK1(), "PUK1")
                and self.is_valid(self.get_PIN2(), "PIN2")
                and self.is_valid(self.get_PUK2(), "PUK2")
                and self.is_valid(self.get_ADM1(), "ADM1")
                and self.is_valid(self.get_ADM6(), "ADM6")
                and self.is_valid(self.get_OP(), "OP")
                and self.is_valid(self.get_K4(), "K4")
                and self.is_valid(self.get_ELECT_DICT(), "DICT")
                and self.is_valid(self.get_GRAPH_DICT(), "DICT")
                # TO DO add server dict here
            )
        return result

    def print_all_global_parameters(self):
        print("PIN1", self.get_PIN1())
        print("PIN2", self.get_PIN2())
        print("PIN2", self.get_PIN2())
        print("DATA SIZE", self.get_DATA_SIZE())


__all__ = ["Parameters", "DataFrames"]


