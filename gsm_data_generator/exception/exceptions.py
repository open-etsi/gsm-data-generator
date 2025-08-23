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


class NoSettingsFilePresent(Exception):
    """No Setting json was found."""


class NoTemplateFilePresent(Exception):
    """No Template json was found."""


class NoJsonFilePresent(Exception):
    """No Template json was found."""


class NoCardError(Exception):
    """No card was found in the reader."""


class ProtocolError(Exception):
    """Some kind of protocol level error interfacing with the card."""


class ReaderError(Exception):
    """Some kind of general error with the card reader."""


class SwMatchError(Exception):
    """Raised when an operation specifies an expected SW but the actual SW from
    the card doesn't match."""

    def __init__(self, sw_actual: str, sw_expected: str, rs=None):
        """
        Args:
                sw_actual : the SW we actually received from the card (4 hex digits)
                sw_expected : the SW we expected to receive from the card (4 hex digits)
                rs : interpreter class to convert SW to string
        """
        self.sw_actual = sw_actual
        self.sw_expected = sw_expected
        self.rs = rs

    def __str__(self):
        if self.rs and self.rs.lchan[0]:
            r = self.rs.lchan[0].interpret_sw(self.sw_actual)
            if r:
                return "SW match failed! Expected %s and got %s: %s - %s" % (
                    self.sw_expected,
                    self.sw_actual,
                    r[0],
                    r[1],
                )
        return "SW match failed! Expected %s and got %s." % (
            self.sw_expected,
            self.sw_actual,
        )


__all__ = [
    "NoSettingsFilePresent",
    "NoTemplateFilePresent",
    "NoJsonFilePresent",
    "NoCardError",
    "ProtocolError",
    "ReaderError",
    "SwMatchError",
]
