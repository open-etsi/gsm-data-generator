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
# pylint: disable=redefined-builtin, wildcard-import
"""DATAGEN: Open Deep Learning Compiler Stack."""
import multiprocessing
import sys
import os

# top-level alias
# gsm_data_generator._ffi
from .base import DATAGENError, __version__  # , #_RUNTIME_ONLY

# from .ffi import register_object, register_func, get_global_func

# # top-level alias
# # gsm_data_generator.runtime
# from .runtime.object import Object
# from .runtime.ndarray import device, cpu, cuda, opencl, vulkan, metal
# from .runtime.ndarray import vpi, rocm, ext_dev, hexagon
# from .runtime import ndarray as nd, DataType, DataTypeCode

# gsm_data_generator.error
from . import error


def _should_print_backtrace():
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ
    gsm_datagen_backtrace = os.environ.get("DATAGEN_BACKTRACE", "0")

    try:
        gsm_datagen_backtrace = bool(int(gsm_datagen_backtrace))
    except ValueError:
        raise ValueError(
            "invalid value for DATAGEN_BACKTRACE {}, please set to 0 or 1.".format(
                gsm_datagen_backtrace
            )
        )

    return in_pytest or gsm_datagen_backtrace


def gsm_datagen_wrap_excepthook(exception_hook):
    """Wrap given excepthook with DATAGEN additional work."""

    def wrapper(exctype, value, trbk):
        """Clean subprocesses when DATAGEN is interrupted."""
        if exctype is error.DiagnosticError and not _should_print_backtrace():
            # TODO(@jroesch): consider moving to C++?
            print(
                "note: run with `DATAGEN_BACKTRACE=1` environment variable to display a backtrace."
            )
        else:
            exception_hook(exctype, value, trbk)

        if hasattr(multiprocessing, "active_children"):
            # pylint: disable=not-callable
            for p in multiprocessing.active_children():
                p.terminate()

    return wrapper


sys.excepthook = gsm_datagen_wrap_excepthook(sys.excepthook)
