# test_utils_json.py
import json
import tempfile
import os
import pytest
from gsm_data_generator.utils import read_json, copy_function


def test_read_json_valid():
    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    tmp_file.write(json.dumps({"key": "value"}))
    tmp_file.close()

    result = read_json(tmp_file.name)
    assert result == {"key": "value"}

    os.remove(tmp_file.name)


def test_copy_function():
    assert copy_function(123) == "123"
    assert copy_function(True) == "True"
    assert copy_function({"a": 1}) == "{'a': 1}"
