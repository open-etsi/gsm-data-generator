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
"""Library information."""


from typing import Tuple

class LibraryInfo:
    __version__ = "0.0.1.dev0"

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

