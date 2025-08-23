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


from ..algorithm import CryptoUtils, DependentDataGenerator
from ..processor import DataProcessing, DataFrameProcessor
from ..globals import DataFrames, Parameters
from ..generator import DataGenerator
from ..utils import copy_function, list_2_dict, DEFAULT_HEADER


class DataGenerationScript:

    def __init__(self, config_holder):
        self.config_holder = config_holder
        self.params = Parameters.get_instance()
        self.dataframes = DataFrames.get_instance()
        self.crypto_utils = CryptoUtils()
        self.data_generator = DataGenerator()
        self.data_processor = DataProcessing()
        self.df_processor = DataFrameProcessor()
        self.dep_data_generator = DependentDataGenerator()

    def json_to_global_params(self) -> None:
        self.params.SERVER_SEP = self.config_holder.DISP.server_data_sep
        self.params.ELECT_SEP = self.config_holder.DISP.elect_data_sep
        self.params.GRAPH_SEP = self.config_holder.DISP.graph_data_sep
        self.params.K4 = self.config_holder.DISP.K4
        self.params.OP = self.config_holder.DISP.op
        self.params.IMSI = self.config_holder.DISP.imsi
        self.params.ICCID = self.config_holder.DISP.iccid
        self.params.PIN1 = self.config_holder.DISP.pin1
        self.params.PUK1 = self.config_holder.DISP.puk1
        self.params.PIN2 = self.config_holder.DISP.pin2
        self.params.PUK2 = self.config_holder.DISP.puk2
        self.params.ADM1 = self.config_holder.DISP.adm1
        self.params.ADM6 = self.config_holder.DISP.adm6
        self.params.DATA_SIZE = self.config_holder.DISP.size

        # self.params.PRODUCTION_CHECK = False

        self.params.ELECT_CHECK = self.config_holder.DISP.elect_check
        self.params.GRAPH_CHECK = self.config_holder.DISP.graph_check
        self.params.SERVER_CHECK = self.config_holder.DISP.server_check

        self.params.ELECT_DICT = list_2_dict(
            self.config_holder.PARAMETERS.data_variables
        )
        self.params.GRAPH_DICT = self.config_holder.PARAMETERS.laser_variables
        self.params.SERVER_DICT = list_2_dict(
            self.config_holder.PARAMETERS.server_variables
        )

        self.params.PIN1_RAND = self.config_holder.DISP.pin1_fix
        self.params.PUK1_RAND = self.config_holder.DISP.puk1_fix
        self.params.PIN2_RAND = self.config_holder.DISP.pin2_fix
        self.params.PUK2_RAND = self.config_holder.DISP.puk2_fix
        self.params.ADM1_RAND = self.config_holder.DISP.adm1_fix
        self.params.ADM6_RAND = self.config_holder.DISP.adm6_fix

    # def json_to_global_params(self):
    #     self.params.set_SERVER_SEP(self.config_holder.DISP.server_data_sep)
    #     self.params.ELECT_SEP =self.config_holder.DISP.elect_data_sep
    #     self.params.set_GRAPH_SEP(self.config_holder.DISP.graph_data_sep)
    #     self.params.set_K4(self.config_holder.DISP.K4)
    #     self.params.set_OP(self.config_holder.DISP.op)
    #     self.params.set_IMSI(self.config_holder.DISP.imsi)
    #     self.params.set_ICCID(self.config_holder.DISP.iccid)
    #     self.params.set_PIN1(self.config_holder.DISP.pin1)
    #     self.params.set_PUK1(self.config_holder.DISP.puk1)
    #     self.params.set_PIN2(self.config_holder.DISP.pin2)
    #     self.params.set_PUK2(self.config_holder.DISP.puk2)
    #     self.params.set_ADM1(self.config_holder.DISP.adm1)
    #     self.params.set_ADM6(self.config_holder.DISP.adm6)
    #     self.params.set_DATA_SIZE(self.config_holder.DISP.size)

    #     #        self.params.set_PRODUCTION_CHECK(False)

    #     self.params.set_ELECT_CHECK(self.config_holder.DISP.elect_check)
    #     self.params.set_GRAPH_CHECK(self.config_holder.DISP.graph_check)
    #     self.params.set_SERVER_CHECK(self.config_holder.DISP.server_check)

    #     self.params.set_ELECT_DICT(
    #         list_2_dict(self.config_holder.PARAMETERS.data_variables)
    #     )
    #     self.params.set_GRAPH_DICT(self.config_holder.PARAMETERS.laser_variables)
    #     self.params.set_SERVER_DICT(
    #         list_2_dict(self.config_holder.PARAMETERS.server_variables)
    #     )

    #     self.params.set_PIN1_RAND(self.config_holder.DISP.pin1_fix)
    #     self.params.set_PUK1_RAND(self.config_holder.DISP.puk1_fix)
    #     self.params.set_PIN2_RAND(self.config_holder.DISP.pin2_fix)
    #     self.params.set_PUK2_RAND(self.config_holder.DISP.puk2_fix)
    #     self.params.set_ADM1_RAND(self.config_holder.DISP.adm1_fix)
    #     self.params.set_ADM6_RAND(self.config_holder.DISP.adm6_fix)

    def generate_eki(self, ki):
        return self.dep_data_generator.calculate_eki(self.params.K4, ki)

    def generate_opc(self, ki):
        return self.dep_data_generator.calculate_opc(self.params.OP, ki)

    def generate_pin(self, pin_type: str) -> str:
        """Generate a PIN: use stored value if *_RAND is True, else random 4-digit."""
        if getattr(self.params, f"{pin_type}_RAND"):
            return getattr(self.params, pin_type)
        return self.data_generator.generate_4_digit()

    def generate_puk(self, puk_type: str) -> str:
        """Generate a PUK: use stored value if *_RAND is True, else random 8-digit."""
        if getattr(self.params, f"{puk_type}_RAND"):
            return getattr(self.params, puk_type)
        return self.data_generator.generate_8_digit()

    def generate_adm(self, adm_type: str) -> str:
        """Generate an ADM: use stored value if *_RAND is True, else random 8-digit."""
        if getattr(self.params, f"{adm_type}_RAND"):
            return getattr(self.params, adm_type)
        return self.data_generator.generate_8_digit()

    # def generate_pin(self, pin_type):
    #     if getattr(self.params, f"{pin_type}_RAND")():
    #         return getattr(self.params, f"{pin_type}")()
    #     return self.data_generator.generate_4_digit()

    # def generate_puk(self, puk_type):
    #     if getattr(self.params, f"{puk_type}_RAND")():
    #         return getattr(self.params, f"{puk_type}")()
    #     return self.data_generator.generate_8_digit()

    # def generate_adm(self, adm_type):
    #     if getattr(self.params, f"{adm_type}_RAND")():
    #         return getattr(self.params, f"{adm_type}")()
    #     return self.data_generator.generate_8_digit()

    def apply_function(self, df, dest: str, src: str, function):
        if dest in df.columns:
            df[dest] = df[src].apply(function).copy(deep=False)

    def apply_functions(self, df):
        df["ICCID"] = df["ICCID"].apply(lambda x: copy_function(x))
        df["IMSI"] = df["IMSI"].apply(lambda x: copy_function(x))
        df["PIN1"] = df["PIN1"].apply(lambda x: self.generate_pin("PIN1"))
        df["PIN2"] = df["PIN2"].apply(lambda x: self.generate_pin("PIN2"))
        df["PUK1"] = df["PUK1"].apply(lambda x: self.generate_puk("PUK1"))
        df["PUK2"] = df["PUK2"].apply(lambda x: self.generate_puk("PUK2"))

        df["ADM1"] = df["ADM1"].apply(lambda x: self.generate_adm("ADM1"))
        df["ADM6"] = df["ADM6"].apply(lambda x: self.generate_adm("ADM6"))

        df["KI"] = df["KI"].apply(lambda x: self.data_generator.generate_ki())
        df["ACC"] = df["IMSI"].apply(
            lambda imsi: self.dep_data_generator.calculate_acc(imsi=str(imsi))
        )

        self.apply_function(df, "EKI", "KI", self.generate_eki)
        self.apply_function(df, "OPC", "KI", self.generate_opc)

        for i in range(1, 4):
            for key in ["KIC", "KID", "KIK"]:
                col = f"{key}{i}"
                if col in df.columns:
                    df[col] = df["KI"].apply(
                        lambda x: self.data_generator.generate_otas()
                    )
        return df

    def generate_demo_data(self):
        df = self.df_processor.generate_empty_dataframe(
            DEFAULT_HEADER, self.params.DATA_SIZE  # type: ignore
        )
        self.df_processor.initialize_column(df, "ICCID", self.params.ICCID)
        self.df_processor.initialize_column(df, "IMSI", self.params.IMSI)
        self.df_processor.initialize_column(df, "OP", self.params.OP, increment=False)
        self.df_processor.initialize_column(df, "K4", self.params.K4, increment=False)
        return self.apply_functions(df)

    def generate_non_demo_data(self):
        input_df = self.dataframes.get_input_df()
        df = self.df_processor.generate_empty_dataframe(DEFAULT_HEADER, len(input_df))  # type: ignore
        self.df_processor.initialize_column(df, "OP", self.params.OP, increment=False)
        self.df_processor.initialize_column(df, "K4", self.params.K4, increment=False)
        df["ICCID"] = input_df["ICCID"]
        df["IMSI"] = input_df["IMSI"]
        return self.apply_functions(df)

    def generate_initial_data(self, is_demo: bool):
        try:
            if is_demo:
                demo_data = self.generate_demo_data()

                k4 = self.params.K4
                op = self.params.OP

                # Validate required parameters
                if not k4 or not isinstance(k4, str):
                    raise ValueError(
                        "Invalid value for K4: must be a non-empty string."
                    )
                if not op or not isinstance(op, str):
                    raise ValueError(
                        "Invalid value for OP: must be a non-empty string."
                    )

                return demo_data, {"k4": k4, "op": op}

            else:
                raise NotImplementedError(
                    "Non-demo data generation is not yet implemented."
                )

        except Exception as e:
            raise RuntimeError(f"Error in generate_initial_data: {e}") from e

    # def generate_initial_data(self, is_demo: bool):
    #     if is_demo:
    #         return self.generate_demo_data(), {
    #             "k4": self.params.get_K4(),
    #             "op": self.params.get_OP(),
    #         }
    #     return self.generate_non_demo_data(), {
    #         "k4": self.params.get_K4(),
    #         "op": self.params.get_OP(),
    #     }

    def process_final_data(
        self, input_dict: dict, df_input: pd.DataFrame, clip: bool, encoding: bool
    ):
        #        print("In process_final_data()", clip, encoding)
        df = df_input.copy(deep=True)
        if encoding:
            df = self.df_processor.encode_dataframe(df)

        headers, _, _, _, left_ranges, right_ranges = (
            self.data_processor.extract_parameter_info(input_dict)
        )
        df = self.df_processor.add_duplicate_columns(df, 10, headers)

        if clip:
            df = self.df_processor.clip_columns(df, left_ranges, right_ranges)
        print(df)
        return df

    # def generate_all_data(self):
    #     # initial_df, keys_dict = self.generate_initial_data(
    #     #     self.params.get_PRODUCTION_CHECK()
    #     # )
    #     initial_df, keys_dict = self.generate_initial_data(True)
    #     print(initial_df.head())
    #     #        data_types = {bool, dict, bool, bool}
    #     data_types = {}
    #     data_types = {
    #         "SERVER": (
    #             self.params.get_SERVER_CHECK(),
    #             self.params.get_SERVER_DICT(),
    #             False,
    #             False,
    #         ),
    #         "GRAPH": (
    #             self.params.get_GRAPH_CHECK(),
    #             self.params.get_GRAPH_DICT(),
    #             True,
    #             False,
    #         ),
    #         "ELECT": (
    #             self.params.get_ELECT_CHECK(),
    #             self.params.get_ELECT_DICT(),
    #             False,
    #             True,
    #         ),
    #     }
    #     result_dfs = {}
    #     for data_type, (check, dict_func, clip, encoding) in data_types.items():
    #         print(data_type, (check, dict_func, clip, encoding))
    #         if check:
    #             result_dfs[data_type] = self.process_final_data(
    #                 dict_func, initial_df, clip, encoding
    #             )
    #     return result_dfs, keys_dict

    def generate_all_data(self):
        # initial_df, keys_dict = self.generate_initial_data(
        #     self.params.get_PRODUCTION_CHECK()
        # )
        initial_df, keys_dict = self.generate_initial_data(True)
        print(initial_df.head())

        data_types = {
            "SERVER": (
                self.params.SERVER_CHECK,
                self.params.SERVER_DICT,
                False,
                False,
            ),
            "GRAPH": (
                self.params.GRAPH_CHECK,
                self.params.GRAPH_DICT,
                True,
                False,
            ),
            "ELECT": (
                self.params.ELECT_CHECK,
                self.params.ELECT_DICT,
                False,
                True,
            ),
        }

        result_dfs = {}

        for data_type, (check, dict_func, clip, encoding) in data_types.items():
            print(f"Checking {data_type}: {check}, {dict_func}")

            if check:
                # 🔹 Raise if dict is missing or invalid
                if not dict_func or not isinstance(dict_func, dict):
                    raise ValueError(
                        f"{data_type} is enabled (check=True) "
                        f"but the dictionary is missing or invalid."
                    )

                try:
                    result_dfs[data_type] = self.process_final_data(
                        dict_func, initial_df, clip, encoding
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed processing {data_type} data: {str(e)}"
                    ) from e

        return result_dfs, keys_dict


__all__ = ["DataGenerationScript"]
