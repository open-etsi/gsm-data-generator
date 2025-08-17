from typing import List, Dict, Any, Tuple
from .gen_utils import EncodingUtils
import pandas as pd
import collections
from .gen_utils import DataTransform


class DataProcessing:

    @staticmethod
    def split_range(input_string: str) -> Tuple[int, int]:
        if input_string and "-" in input_string and len(input_string) > 2:
            values = input_string.split("-")
            return int(values[0]), int(values[1])
        return 0, 32

    @staticmethod
    def extract_ranges(ranges: List[str]) -> Tuple[List[int], List[int]]:
        left_ranges, right_ranges = [], []
        for range_str in ranges:
            left, right = DataProcessing.split_range(range_str)
            left_ranges.append(left)
            right_ranges.append(right)
        return left_ranges, right_ranges

    @staticmethod
    def find_duplicates(items: List[Any]) -> List[Any]:
        return [item for item, count in collections.Counter(items).items() if count > 1]

    @staticmethod
    def extract_parameter_info(
        param_dict: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str], set, List[str], List[int], List[int]]:
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
    @staticmethod
    def generate_empty_dataframe(columns: List[str], rows: str) -> pd.DataFrame:
        empty_data = [{col: 0 for col in columns} for _ in range(int(rows))]
        return pd.DataFrame(empty_data)

    @staticmethod
    def initialize_column(
        df: pd.DataFrame, column: str, start_value: str, increment: bool = True
    ) -> None:
        if increment:
            df[column] = range(int(start_value), int(start_value) + len(df))
        else:
            df[column] = str(start_value)

    @staticmethod
    def apply_function_to_column(
        df: pd.DataFrame, dest_col: str, src_col: str, func
    ) -> None:
        if dest_col in df.columns:
            df[dest_col] = df[src_col].apply(func)

    @staticmethod
    def clip_columns(
        df: pd.DataFrame, left_ranges: List[int], right_ranges: List[int]
    ) -> pd.DataFrame:
        for col, left, right in zip(df.columns, left_ranges, right_ranges):
            df[col] = df[col].apply(lambda x: x[left : right + 1])
        return df

    @staticmethod
    def add_duplicate_columns(
        df: pd.DataFrame, limit: int, headers: List[str]
    ) -> pd.DataFrame:
        for c in range(limit):
            for col in df.columns:
                new_col = f"{col}{c}"
                if new_col in headers:
                    df[new_col] = df[col]
        return df[headers]

    @staticmethod
    def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
            if col in df.columns:
                df[col] = df[col].apply(func)
        return df

    @staticmethod
    def decode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
            if col in df.columns:
                df[col] = df[col].apply(func)
        return df


__all__ = [
    "DataProcessing",
    "DataFrameProcessor",
]
