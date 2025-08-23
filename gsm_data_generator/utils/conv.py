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


def list_2_dict(list: list) -> dict:
    dict = {}
    for index in range(0, len(list)):
        dict[str(index)] = [list[index], "Normal", "0-31"]
    return dict


def dict_2_list(d: dict) -> list:
    list1 = []
    for index, j in enumerate(d):
        temp = list(d.values())[index][0]
        list1.append(temp)
    return list1


__all__ = ["list_2_dict", "dict_2_list"]
