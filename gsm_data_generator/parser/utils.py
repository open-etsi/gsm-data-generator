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

from pydantic import BaseModel, Field, conint, constr
from dataclasses import dataclass
from typing import List, Dict
import json


class DISP(BaseModel):
    elect_data_sep: str = Field(..., min_length=1)
    server_data_sep: str = Field(..., min_length=1)
    graph_data_sep: str = Field(..., min_length=1)
    K4: constr(min_length=32, max_length=64)  # type: ignore
    op: constr(min_length=32, max_length=32)  # type: ignore
    imsi: constr(min_length=15, max_length=15)  # type: ignore
    iccid: constr(min_length=18, max_length=19)  # type: ignore
    pin1: constr(min_length=4, max_length=4)  # type: ignore
    puk1: constr(min_length=8, max_length=8)  # type: ignore
    pin2: constr(min_length=4, max_length=4)  # type: ignore
    puk2: constr(min_length=8, max_length=8)  # type: ignore
    adm1: constr(min_length=8, max_length=8)  # type: ignore
    adm6: constr(min_length=8, max_length=8)  # type: ignore
    size: conint(ge=1, le=1000000)  # type: ignore
    prod_check: bool
    elect_check: bool
    graph_check: bool
    server_check: bool
    pin1_fix: bool
    puk1_fix: bool
    pin2_fix: bool
    puk2_fix: bool
    adm1_fix: bool
    adm6_fix: bool


class PATHS(BaseModel):
    FILE_NAME: str
    OUTPUT_FILES_DIR: str
    OUTPUT_FILES_LASER_EXT: str


class PARAMETERS(BaseModel):
    server_variables: List[str]
    data_variables: List[str]
    laser_variables: Dict[str, List[str]]


class ConfigData(BaseModel):
    DISP: DISP
    PATHS: PATHS
    PARAMETERS: PARAMETERS


@dataclass
class ConfigHolder:
    DISP: DISP
    PATHS: PATHS
    PARAMETERS: PARAMETERS

    @classmethod
    def from_config(cls, config: ConfigData):
        return cls(DISP=config.DISP, PATHS=config.PATHS, PARAMETERS=config.PARAMETERS)


def json_loader(path: str) -> ConfigHolder:
    with open(path, "r") as f:
        data = json.load(f)
    config = ConfigData(**data)
    config_holder = ConfigHolder.from_config(config)
    return config_holder


def json_loader_2_ConfigHolder(input_data: dict | str) -> ConfigHolder:
    if isinstance(input_data, str):
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON string provided.") from e
    elif isinstance(input_data, dict):
        data = input_data
    else:
        raise ValueError("Input must be a dictionary or JSON string.")

    # Validate and create ConfigData instance
    config = ConfigData(**data)
    # Create and return ConfigHolder instance
    return ConfigHolder.from_config(config)


def gui_loader(path) -> ConfigHolder:
    #    data = json.load(path)
    data = path
    config = ConfigData(**data)
    config_holder = ConfigHolder.from_config(config)
    return config_holder


# # if __name__ == "__main__":
# config_holder = json_loader(
#     "D:\STC_APP\improvements\security-layer\datageneration\core\settings.json"
# )

# # Load config data
# with open(
#     "D:\STC_APP\improvements\security-layer\datageneration\core\settings.json", "r"
# ) as f:
#     data = json.load(f)
# config = ConfigData(**data)
# config_holder = ConfigHolder.from_config(config)

__all__ = [
    "DISP",
    "PATHS",
    "PARAMETERS",
    "ConfigHolder",
    "json_loader",
    "json_loader_2_ConfigHolder",
    "gui_loader",
    "ConfigData",
]
