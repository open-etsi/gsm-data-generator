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

"""Data Processing"""
from typing import List, Dict, Any, Tuple
import collections
import pandas as pd
from ..transform import DataTransform
from ..algorithm import EncodingUtils


class DataProcessing:
    """Class providing utility methods for processing and transforming data."""

    @staticmethod
    def split_range(input_string: str) -> Tuple[int, int]:
        """Split a range string like '0-32' into a tuple of integers (start, end)."""

        if input_string and "-" in input_string and len(input_string) > 2:
            values = input_string.split("-")
            return int(values[0]), int(values[1])
        return 0, 32

    @staticmethod
    def extract_ranges(ranges: List[str]) -> Tuple[List[int], List[int]]:
        """Extract left and right integer values from a list of range strings."""

        left_ranges, right_ranges = [], []
        for range_str in ranges:
            left, right = DataProcessing.split_range(range_str)
            left_ranges.append(left)
            right_ranges.append(right)
        return left_ranges, right_ranges

    @staticmethod
    def find_duplicates(items: List[Any]) -> List[Any]:
        """Return a list of duplicate items from the given list."""

        return [item for item, count in collections.Counter(items).items() if count > 1]

    @staticmethod
    def extract_parameter_info(
        param_dict: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str], set, List[str], List[int], List[int]]:
        """
        Extract detailed information from a parameter dictionary, including renamed values,
        duplicates, unique items, classes, and range boundaries.
        """
        values, classes, ranges = [], [], []
        for item in param_dict.values():
            values.append(item[0])
            classes.append(item[1])
            ranges.append(item[2])

        renamed_values = DataProcessing.append_count_to_duplicates(values)
        duplicate_values = DataProcessing.find_duplicates(values)
        unique_values = set(values)
        left_ranges, right_ranges = DataProcessing.extract_ranges(ranges)

        return (
            renamed_values,
            duplicate_values,
            unique_values,
            classes,
            left_ranges,
            right_ranges,
        )

    @staticmethod
    def append_count_to_duplicates(input_list: List[str]) -> List[str]:
        """Append count suffix to duplicate elements in a list to make them unique."""

        output_list = []
        element_counts: Dict[str, int] = {}

        for element in input_list:
            if element in element_counts:
                element_counts[element] += 1
                output_list.append(f"{element}{element_counts[element]}")
            else:
                element_counts[element] = 0
                output_list.append(element)

        return output_list


class DataFrameProcessor:
    """Class providing static methods to manipulate and encode pandas DataFrames."""

    @staticmethod
    def generate_empty_dataframe(columns: List[str], rows: str) -> pd.DataFrame:
        """Generate an empty DataFrame with specified columns and number of rows."""

        empty_data = [{col: 0 for col in columns} for _ in range(int(rows))]
        return pd.DataFrame(empty_data)

    @staticmethod
    def initialize_column(
        dataframe: pd.DataFrame, column: str, start_value: str, increment: bool = True
    ) -> None:
        """Initialize a DataFrame column with either a range of integers or a constant value."""

        if increment:
            dataframe[column] = range(
                int(start_value), int(start_value) + len(dataframe)
            )
        else:
            dataframe[column] = str(start_value)

    @staticmethod
    def apply_function_to_column(
        dataframe: pd.DataFrame, dest_col: str, src_col: str, func
    ) -> None:
        """Apply a function to a source column and store the result in a destination column."""

        if dest_col in dataframe.columns:
            dataframe[dest_col] = dataframe[src_col].apply(func)

    @staticmethod
    def clip_columns(
        dataframe: pd.DataFrame, left_ranges: List[int], right_ranges: List[int]
    ) -> pd.DataFrame:
        """Clip values of each column in a DataFrame based on provided left and right indices."""

        for col, left, right in zip(dataframe.columns, left_ranges, right_ranges):
            dataframe[col] = dataframe[col].apply(lambda x: x[left : right + 1])
        return dataframe

    @staticmethod
    def add_duplicate_columns(
        dataframe: pd.DataFrame, limit: int, headers: List[str]
    ) -> pd.DataFrame:
        """Duplicate columns in the DataFrame up to a given limit, updating headers accordingly."""

        for c in range(limit):
            for col in dataframe.columns:
                new_col = f"{col}{c}"
                if new_col in headers:
                    dataframe[new_col] = dataframe[col]
        return dataframe[headers]

    @staticmethod
    def encode_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Encode specific DataFrame columns using predefined encoding functions."""

        encoding_map = {
            "ICCID": EncodingUtils.enc_iccid,
            "IMSI": EncodingUtils.enc_imsi,
            "PIN1": EncodingUtils.enc_pin,
            "PUK1": DataTransform.s2h,
            "PIN2": EncodingUtils.enc_pin,
            "PUK2": DataTransform.s2h,
            "ADM1": DataTransform.s2h,
            "ADM6": DataTransform.s2h,
        }
        for col, func in encoding_map.items():
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].apply(func)
        return dataframe

    @staticmethod
    def decode_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Decode specific DataFrame columns using predefined decoding functions."""

        decoding_map = {
            "ICCID": EncodingUtils.dec_iccid,
            "IMSI": EncodingUtils.dec_imsi,
            "PIN1": EncodingUtils.dec_pin,
            "PUK1": DataTransform.h2s,
            "PIN2": EncodingUtils.dec_pin,
            "PUK2": DataTransform.h2s,
            "ADM1": DataTransform.h2s,
        }
        for col, func in decoding_map.items():
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].apply(func)
        return dataframe


__all__ = [
    "DataProcessing",
    "DataFrameProcessor",
]
