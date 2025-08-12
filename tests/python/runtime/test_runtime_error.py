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
"""Test runtime error handling"""

import functools
import platform
import subprocess
import traceback

import pytest

import gsm_data_generator
import gsm_data_generator.testing


def test_op_translation_to_not_implemented():
    try:
        gsm_data_generator.testing.test_raise_error("OpNotImplemented", "myop")
        assert False
    except gsm_data_generator.error.OpNotImplemented as e:
        assert isinstance(e, NotImplementedError)


def test_op_translation_to_internal_error():
    fchk_eq = gsm_data_generator.testing.test_check_eq_callback("InternalError: myop")
    try:
        fchk_eq(0, 1)
        assert False
    except gsm_data_generator.error.InternalError as e:
        pass


def test_op_translation_to_value_error():
    try:
        gsm_data_generator.testing.ErrorTest(0, 1)
        assert False
    except ValueError as e:
        pass


def test_deep_callback():
    """Propagate python errors through API calls

    If a Python exception is raised, and that exception is caught in
    Python, the original exception should be propagated so that the
    traceback contains all intermediate python frames.

    Stack
    - test_deep_callback
    - test

    """

    def error_callback():
        raise ValueError("callback error")

    wrap1 = gsm_data_generator.testing.test_wrap_callback(error_callback)

    def flevel2():
        wrap1()

    wrap2 = gsm_data_generator.testing.test_wrap_callback(flevel2)

    def flevel3():
        wrap2()

    wrap3 = gsm_data_generator.testing.test_wrap_callback(flevel3)

    try:
        wrap3()
        assert False
    except ValueError as err:
        frames = traceback.extract_tb(err.__traceback__)

    local_frames = [frame.name for frame in frames if frame.filename == __file__]
    assert local_frames == ["test_deep_callback", "flevel3", "flevel2", "error_callback"]


@functools.lru_cache()
def _has_debug_symbols():
    lib = gsm_data_generator.base._LIB
    headers = subprocess.check_output(["objdump", "--section-headers", lib._name], encoding="utf-8")
    return ".debug" in headers


@pytest.mark.skipif(
    not _has_debug_symbols() or platform.machine != "x86_64",
    reason="C++ stack frames require debug symbols, only implemented for x86",
)
def test_cpp_frames_in_stack_trace_from_python_error():
    """A python exception crossing C++ boundaries should have C++ stack frames"""

    def error_callback():
        raise ValueError("callback error")

    wrapped = gsm_data_generator.testing.test_wrap_callback(error_callback)

    try:
        wrapped()
        assert False
    except ValueError as err:
        frames = traceback.extract_tb(err.__traceback__)

        cpp_frames = [
            frame
            for frame in frames
            if frame.filename.endswith(".cc") or frame.filename.endswith(".c")
        ]
        assert len(cpp_frames) >= 1, (
            f"Traceback through files '{[frame.filename for frame in frames]}'"
            f" expected to contain C/C++ frames, "
            f" but instead caught exception {err}"
        )


@pytest.mark.skipif(
    not _has_debug_symbols() or platform.machine != "x86_64",
    reason="C++ stack frames require debug symbols, only implemented for x86",
)
def test_stack_trace_from_cpp_error():
    """A python exception originating in C++ should have C++ stack frames"""
    try:
        gsm_data_generator.testing.ErrorTest(0, 1)
        assert False
    except ValueError as err:
        frames = traceback.extract_tb(err.__traceback__)

        cpp_frames = [
            frame
            for frame in frames
            if frame.filename.endswith(".cc") or frame.filename.endswith(".c")
        ]
        assert len(cpp_frames) >= 1, (
            f"Traceback through files '{[frame.filename for frame in frames]}'"
            f" expected to contain C/C++ frames, "
            f" but instead caught exception {err}"
        )


if __name__ == "__main__":
    gsm_data_generator.testing.main()
