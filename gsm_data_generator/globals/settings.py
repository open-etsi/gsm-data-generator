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

# from PyQt6.QtCore import QSettings
# from PyQt6.QtWidgets import QWidget
# from .parameters import Parameters


# class SETTINGS(QWidget):
#     __instance = None

#     def __init__(self, ui):
#         """
#         GET and SET settings for GUI
#         """
#         super().__init__()
#         self.ui = ui
#         if SETTINGS.__instance is not None:
#             raise Exception(
#                 "SETTINGS class is a singleton! Use get_instance() to access the instance."
#             )
#         else:
#             SETTINGS.__instance = self
#         self.parameters = Parameters.get_instance()
#         self.IMSI_SETT = QSettings("IMSI", "")
#         self.ICCID_SETT = QSettings("ICCID", "")
#         self.PIN1_SETT = QSettings("PIN1", "")
#         self.PUK1_SETT = QSettings("PUK1", "")
#         self.PIN2_SETT = QSettings("PIN2", "")
#         self.PUK2_SETT = QSettings("PUK2", "")
#         self.ADM1_SETT = QSettings("ADM1", "")
#         self.ADM6_SETT = QSettings("ADM6", "")
#         self.ACC_SETT = QSettings("ACC", "")
#         self.K4_SETT = QSettings("K4", "")
#         self.OP_SETT = QSettings("OP", "")
#         self.DATA_SIZE_SETT = QSettings("DATA_SIZE", "")

#         self.PIN1_RAND_CHECK_SETT = QSettings("PIN1_RAND", "")
#         self.PIN2_RAND_CHECK_SETT = QSettings("PIN2_RAND", "")
#         self.PUK1_RAND_CHECK_SETT = QSettings("PUK1_RAND", "")
#         self.PUK2_RAND_CHECK_SETT = QSettings("PUK2_RAND", "")
#         self.ADM1_RAND_CHECK_SETT = QSettings("ADM1_RAND", "")
#         self.ADM6_RAND_CHECK_SETT = QSettings("ADM6_RAND", "")

#     # @staticmethod
#     # def get_instance():
#     #     if SETTINGS.__instance is None:
#     #         SETTINGS.__instance = SETTINGS()
#     #     return SETTINGS.__instance

#     def __del__(self):
#         self.save_global_params_to_settings()

#     @staticmethod
#     def BoolToValue(value: bool):
#         if isinstance(value, bool):
#             if value:
#                 return "True"
#             else:
#                 return "False"
#         else:
#             return value

#     @staticmethod
#     def valueToBool(value: str):
#         if isinstance(value, str):
#             if value.upper() == "TRUE":
#                 return True
#             else:
#                 return False
#         else:
#             return value

#     def set_gui_from_settings(self):
#         self.ui.imsi_text.setText(self.parameters.get_IMSI())
#         self.ui.iccid_text.setText(self.parameters.get_ICCID())
#         self.ui.pin1_text.setText(self.parameters.get_PIN1())
#         self.ui.puk1_text.setText(self.parameters.get_PUK1())
#         self.ui.pin2_text.setText(self.parameters.get_PIN2())
#         self.ui.puk2_text.setText(self.parameters.get_PUK2())
#         self.ui.adm1_text.setText(self.parameters.get_ADM1())
#         self.ui.adm6_text.setText(self.parameters.get_ADM6())
#         self.ui.k4_key_text.setText(self.parameters.get_K4())
#         self.ui.op_key_text.setText(self.parameters.get_OP())
#         self.ui.data_size_text.setText(self.parameters.get_DATA_SIZE())
#         self.ui.pin1_rand_check.setChecked(bool(self.parameters.get_PIN1_RAND()))
#         self.ui.pin2_rand_check.setChecked(bool(self.parameters.get_PIN2_RAND()))
#         self.ui.puk1_rand_check.setChecked(bool(self.parameters.get_PUK1_RAND()))
#         self.ui.puk2_rand_check.setChecked(bool(self.parameters.get_PUK2_RAND()))
#         self.ui.adm1_rand_check.setChecked(bool(self.parameters.get_ADM1_RAND()))
#         self.ui.adm6_rand_check.setChecked(bool(self.parameters.get_ADM6_RAND()))

#     def save_global_params_to_settings(self):
#         self.IMSI_SETT.setValue("IMSI", self.parameters.get_IMSI())
#         self.ICCID_SETT.setValue("ICCID", self.parameters.get_ICCID())
#         self.PIN1_SETT.setValue("PIN1", self.parameters.get_PIN1())
#         self.PUK1_SETT.setValue("PUK1", self.parameters.get_PUK1())
#         self.PIN2_SETT.setValue("PIN2", self.parameters.get_PIN2())
#         self.PUK2_SETT.setValue("PUK2", self.parameters.get_PUK2())
#         self.ADM1_SETT.setValue("ADM1", self.parameters.get_ADM1())
#         self.ADM6_SETT.setValue("ADM6", self.parameters.get_ADM6())
#         self.K4_SETT.setValue("K4", self.parameters.get_K4())
#         self.OP_SETT.setValue("OP", self.parameters.get_OP())
#         self.DATA_SIZE_SETT.setValue("DATA_SIZE", self.parameters.get_DATA_SIZE())

#         self.PIN1_RAND_CHECK_SETT.setValue(
#             "PIN1_RAND", self.BoolToValue(self.parameters.get_PIN1_RAND())
#         )
#         self.PIN2_RAND_CHECK_SETT.setValue(
#             "PIN2_RAND", self.BoolToValue(self.parameters.get_PIN2_RAND())
#         )
#         self.PUK1_RAND_CHECK_SETT.setValue(
#             "PUK1_RAND", self.BoolToValue(self.parameters.get_PUK1_RAND())
#         )
#         self.PUK2_RAND_CHECK_SETT.setValue(
#             "PUK2_RAND", self.BoolToValue(self.parameters.get_PUK2_RAND())
#         )
#         self.ADM1_RAND_CHECK_SETT.setValue(
#             "ADM1_RAND", self.BoolToValue(self.parameters.get_ADM1_RAND())
#         )
#         self.ADM6_RAND_CHECK_SETT.setValue(
#             "ADM6_RAND", self.BoolToValue(self.parameters.get_ADM6_RAND())
#         )

#     def load_settings_to_global(self):
#         self.parameters.set_IMSI(self.IMSI_SETT.value("IMSI"))
#         self.parameters.set_ICCID(self.ICCID_SETT.value("ICCID"))
#         self.parameters.set_PIN1(self.PIN1_SETT.value("PIN1"))
#         self.parameters.set_PUK1(self.PUK1_SETT.value("PUK1"))
#         self.parameters.set_PIN2(self.PIN2_SETT.value("PIN2"))
#         self.parameters.set_PUK2(self.PUK2_SETT.value("PUK2"))
#         self.parameters.set_ADM1(self.ADM1_SETT.value("ADM1"))
#         self.parameters.set_ADM6(self.ADM6_SETT.value("ADM6"))
#         #        self.parameters.set_K4(self.K4_SETT.value("K4"))
#         #        self.parameters.set_OP(self.OP_SETT.value("OP"))
#         self.parameters.set_DATA_SIZE(self.DATA_SIZE_SETT.value("DATA_SIZE"))

#         self.parameters.set_PIN1_RAND(
#             self.valueToBool(self.PIN1_RAND_CHECK_SETT.value("PIN1_RAND"))
#         )
#         self.parameters.set_PIN2_RAND(
#             self.valueToBool(self.PIN2_RAND_CHECK_SETT.value("PIN2_RAND"))
#         )
#         self.parameters.set_PUK1_RAND(
#             self.valueToBool(self.PUK1_RAND_CHECK_SETT.value("PUK1_RAND"))
#         )
#         self.parameters.set_PUK2_RAND(
#             self.valueToBool(self.PUK2_RAND_CHECK_SETT.value("PUK2_RAND"))
#         )
#         self.parameters.set_ADM1_RAND(
#             self.valueToBool(self.ADM1_RAND_CHECK_SETT.value("ADM1_RAND"))
#         )
#         self.parameters.set_ADM6_RAND(
#             self.valueToBool(self.ADM6_RAND_CHECK_SETT.value("ADM6_RAND"))
#         )


# # s=SETTINGS()
# # s.save_settings()
# # s.load_settings()

# __all__ = ["SETTINGS"]
