import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from gsm_data_generator.executor import DataGenerationScript

# ----------------------------
# FIXTURES
# ----------------------------

# @pytest.fixture
# def mock_config():
#     """Mock config_holder with DISP + PARAMETERS attributes."""
#     class DISP:
#         server_data_sep = "|"
#         elect_data_sep = ","
#         graph_data_sep = ";"
#         K4, op, imsi, iccid = "K4VAL", "OPVAL", "IMSI001", "ICCID001"
#         pin1, puk1, pin2, puk2 = "1111", "22222222", "3333", "44444444"
#         adm1, adm6 = "55555555", "66666666"
#         size = 2
#         elect_check, graph_check, server_check = True, True, True
#         pin1_fix, puk1_fix, pin2_fix, puk2_fix, adm1_fix, adm6_fix = False, False, False, False, False, False

#     class PARAMETERS:
#         data_variables = ["d1", "d2"]
#         laser_variables = {"l1": 1}
#         server_variables = ["s1", "s2"]

#     return MagicMock(DISP=DISP, PARAMETERS=PARAMETERS)

# @pytest.fixture
# def script(mock_config):
#     """Return a DataGenerationScript with patched dependencies."""
#     with patch("gsm_data_generator.globals") as MockParams, \
#          patch("gsm_data_generator.globals") as MockDFs:

#         # Mock Parameters singleton
#         mock_params = MagicMock()
#         mock_params.get_K4.return_value = "K4VAL"
#         mock_params.get_OP.return_value = "OPVAL"
#         mock_params.get_IMSI.return_value = "IMSI001"
#         mock_params.get_ICCID.return_value = "ICCID001"
#         mock_params.get_DATA_SIZE.return_value = 2
#         MockParams.get_instance.return_value = mock_params

#         # Mock DataFrames singleton
#         mock_dfs = MagicMock()
#         mock_dfs.get_input_df.return_value = pd.DataFrame(
#             {"ICCID": ["I1", "I2"], "IMSI": ["M1", "M2"]}
#         )
#         MockDFs.get_instance.return_value = mock_dfs

#         # Return script instance
#         return DataGenerationScript(config_holder=mock_config, params=mock_params, dataframes=mock_dfs)


# ----------------------------
# TESTS
# ----------------------------

# def test_json_to_global_params(script, mock_config):
#     script.json_to_global_params()
#     # Verify DISP values mapped correctly
#     script.params.set_SERVER_SEP.assert_called_once_with(mock_config.DISP.server_data_sep)
#     script.params.set_ELECT_DICT.assert_called_once()


# def test_generate_code_fixed(script):
#     script.params.get_PIN1_RAND.return_value = True
#     script.params.get_PIN1.return_value = "9999"
#     assert script.generate_code("PIN1", 4) == "9999"


# def test_generate_code_random(script):
#     script.params.get_PIN1_RAND.return_value = False
#     script.data_generator.generate_4_digit = MagicMock(return_value="1234")
#     assert script.generate_code("PIN1", 4) == "1234"


# def test_apply_function(script):
#     df = pd.DataFrame({"KI": ["aaa"], "EKI": [""]})
#     df_out = script.apply_function(df, "EKI", "KI", lambda x: x.upper())
#     assert df_out["EKI"].iloc[0] == "AAA"


# def test_apply_functions_generates_columns(script):
#     # Mock generators
#     script.data_generator.generate_4_digit = MagicMock(return_value="1111")
#     script.data_generator.generate_8_digit = MagicMock(return_value="22222222")
#     script.data_generator.generate_ki = MagicMock(return_value="KI001")
#     script.data_generator.generate_otas = MagicMock(return_value="OTAS1")
#     script.dep_data_generator.calculate_acc = MagicMock(return_value="ACC1")
#     script.generate_eki = MagicMock(return_value="EKI001")
#     script.generate_opc = MagicMock(return_value="OPC001")

#     df = pd.DataFrame({
#         "ICCID": ["iccid"],
#         "IMSI": ["imsi"],
#         "PIN1": [""], "PIN2": [""],
#         "PUK1": [""], "PUK2": [""],
#         "ADM1": [""], "ADM6": [""],
#         "KI": [""],
#         "ACC": [""],
#         "EKI": [""], "OPC": [""],
#         "KIC1": [""], "KID1": [""], "KIK1": [""],
#     })

#     df_out = script.apply_functions(df)

#     assert df_out["PIN1"].iloc[0] == "1111"
#     assert df_out["PUK1"].iloc[0] == "22222222"
#     assert df_out["KI"].iloc[0] == "KI001"
#     assert df_out["EKI"].iloc[0] == "EKI001"
#     assert df_out["OPC"].iloc[0] == "OPC001"
#     assert df_out["KIC1"].iloc[0] == "OTAS1"


# def test_generate_demo_data(script):
#     script.df_processor.generate_empty_dataframe = MagicMock(
#         return_value=pd.DataFrame({"ICCID": [""], "IMSI": [""], "KI": [""], "ACC": [""], "EKI": [""], "OPC": [""]})
#     )
#     script.df_processor.initialize_column = MagicMock()

#     script.apply_functions = MagicMock(return_value="DF_OUT")
#     df_out = script.generate_demo_data()
#     assert df_out == "DF_OUT"


# def test_generate_non_demo_data(script):
#     script.df_processor.generate_empty_dataframe = MagicMock(
#         return_value=pd.DataFrame({"ICCID": [""], "IMSI": [""], "KI": [""], "ACC": [""], "EKI": [""], "OPC": [""]})
#     )
#     script.df_processor.initialize_column = MagicMock()
#     script.apply_functions = MagicMock(return_value="DF_OUT")

#     df_out = script.generate_non_demo_data()
#     assert df_out == "DF_OUT"


# def test_generate_initial_data_demo(script):
#     script.generate_demo_data = MagicMock(return_value="DEMO_DF")
#     df, keys = script.generate_initial_data(is_demo=True)
#     assert df == "DEMO_DF"
#     assert "k4" in keys and "op" in keys


# def test_process_final_data_with_encoding_and_clip(script):
#     # Mock df
#     df_input = pd.DataFrame({"A": [1, 2]})
#     script.df_processor.encode_dataframe = MagicMock(return_value=df_input)
#     script.data_processor.extract_parameter_info = MagicMock(
#         return_value=(["A"], None, None, None, [0], [1])
#     )
#     script.df_processor.add_duplicate_columns = MagicMock(return_value=df_input)
#     script.df_processor.clip_columns = MagicMock(return_value="CLIPPED_DF")

#     result = script.process_final_data({}, df_input, clip=True, encoding=True)
#     assert result == "CLIPPED_DF"
