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

import gsm_data_generator
import gsm_data_generator.testing

from gsm_data_generator.script import tir as T, ir as I


def _device_context(dev_type, dev_id):
    ctx = gsm_data_generator.tir.call_extern("handle", "device_context", dev_type, dev_id)
    return gsm_data_generator.tir.Call("handle", "tir.tvm_thread_context", [ctx])


class TestCombineContextsInLoop(gsm_data_generator.testing.CompareBeforeAfter):
    """Device contexts should be hoisted and merged"""

    transform = gsm_data_generator.tir.transform.CombineContextCall()

    def before(self):
        @T.prim_func
        def func(dev_type: T.int32, n: T.int32):
            T.func_attr({"target": T.target("llvm")})
            A = T.allocate([n], "float32", "global")
            for i in range(n):
                T.call_extern(
                    "int32",
                    "fadd",
                    _device_context(dev_type, 0),
                    A,
                )
                for j in range(10):
                    T.call_extern(
                        "int32",
                        "fadd",
                        _device_context(dev_type, 1),
                        A,
                    )
                    T.call_extern(
                        "int32",
                        "fadd",
                        _device_context(dev_type, 0),
                        A,
                    )

        return func

    def expected(dev_type: T.int32, n: T.int32):
        T.func_attr({"target": T.target("llvm")})
        ctx_cache_: T.handle = T.call_extern("handle", "device_context", dev_type, 0)
        ctx_cache__1: T.handle = T.call_extern("handle", "device_context", dev_type, 1)
        A = T.allocate([n], "float32", "global")
        for i in range(n):
            T.call_extern("int32", "fadd", ctx_cache_, A)
            for j in range(10):
                T.call_extern("int32", "fadd", ctx_cache__1, A)
                T.call_extern("int32", "fadd", ctx_cache_, A)


class TestCombineContextsInLoopWithoutTarget(TestCombineContextsInLoop):
    """CombineContextCall only updates host-side functions"""

    def before(self):
        @T.prim_func
        def func(dev_type: T.int32, n: T.int32):
            A = T.allocate([n], "float32", "global")
            for i in range(n):
                T.call_extern(
                    "int32",
                    "fadd",
                    _device_context(dev_type, 0),
                    A,
                )
                for j in range(10):
                    T.call_extern(
                        "int32",
                        "fadd",
                        _device_context(dev_type, 1),
                        A,
                    )
                    T.call_extern(
                        "int32",
                        "fadd",
                        _device_context(dev_type, 0),
                        A,
                    )

        return func

    expected = before


if __name__ == "__main__":
    gsm_data_generator.testing.main()
