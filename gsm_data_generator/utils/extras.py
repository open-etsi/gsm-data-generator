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

import json


# TO DO it using context maneger : use magic methods etc
# Create class this

# class CustomOpen(object):
#     def __init__(self, filename):
#         self.file = open(filename)

#     def __enter__(self):
#         return self.file

#     def __exit__(self, ctx_type, ctx_value, ctx_traceback):
#         self.file.close()

# with CustomOpen('file') as f:
#     contents = f.read()


def read_json(file_path: str):
    """
    Read a JSON file from the given file path and return its contents as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data as a dictionary, or None if the file is not found
        or cannot be decoded.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return dict(data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{file_path}': {e}")
        return None


def copy_function(x):
    """
    Convert the input to its string representation.

    Args:
        x: Any input value.

    Returns:
        str: String representation of the input.
    """
    return str(x)
