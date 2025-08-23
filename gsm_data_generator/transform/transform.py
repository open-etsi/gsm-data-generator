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

from typing import List


class DataTransform:
    @staticmethod
    def swap_nibbles(s: str) -> str:
        return "".join([x + y for x, y in zip(s[1::2], s[0::2])])

    @staticmethod
    def rpad(s: str, l: int, c: str = "f") -> str:
        return s + c * (l - len(s))

    @staticmethod
    def lpad(s: str, l: int, c: str = "f") -> str:
        return c * (l - len(s)) + s

    @staticmethod
    def half_round_up(n: int) -> int:
        return (n + 1) // 2

    @staticmethod
    def h2b(s: str) -> bytearray:
        return bytearray.fromhex(s)

    @staticmethod
    def b2h(b: bytearray) -> str:
        return "".join([f"{x:02x}" for x in b])

    @staticmethod
    def h2i(s: str) -> List[int]:
        return [(int(x, 16) << 4) + int(y, 16) for x, y in zip(s[0::2], s[1::2])]

    @staticmethod
    def i2h(s: List[int]) -> str:
        return "".join([f"{x:02x}" for x in s])

    @staticmethod
    def h2s(s: str) -> str:
        return "".join(
            [
                chr((int(x, 16) << 4) + int(y, 16))
                for x, y in zip(s[0::2], s[1::2])
                if int(x + y, 16) != 0xFF
            ]
        )

    @staticmethod
    def s2h(s: str) -> str:
        return DataTransform.b2h(bytearray(map(ord, s)))

    @staticmethod
    def i2s(s: List[int]) -> str:
        return "".join([chr(x) for x in s])


__all__ = ["DataTransform"]
