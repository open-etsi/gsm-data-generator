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
from gsm_data_generator import te
from gsm_data_generator.script import tir as T
import os


def test_unroll_loop():
    ib = gsm_data_generator.tir.ir_builder.create()
    dtype = "int64"
    n = te.size_var("n")
    Ab = gsm_data_generator.tir.decl_buffer((n,), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(n, n + 2, name="i") as i:
        with ib.for_range(0, 8, name="i", kind="unroll") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()
    mod = gsm_data_generator.IRModule.from_expr(gsm_data_generator.tir.PrimFunc([Ab], stmt))

    assert isinstance(stmt, gsm_data_generator.tir.For)

    with gsm_data_generator.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        ret = gsm_data_generator.tir.transform.UnrollLoop()(mod)["main"].body
        assert not isinstance(ret, gsm_data_generator.tir.For)

    with gsm_data_generator.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 15}}):
        ret = gsm_data_generator.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret, gsm_data_generator.tir.For)

    with gsm_data_generator.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_step": 16, "explicit_unroll": False}}
    ):
        ret = gsm_data_generator.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret, gsm_data_generator.tir.For)
        assert ret.kind == gsm_data_generator.tir.ForKind.UNROLLED

    ib = gsm_data_generator.tir.ir_builder.create()
    ib.scope_attr(gsm_data_generator.tir.const(0, "int32"), "pragma_auto_unroll_max_step", 16)
    ib.emit(stmt)
    wrapped = ib.get()
    wrapped = gsm_data_generator.tir.SeqStmt([wrapped, stmt])
    assert isinstance(ret, gsm_data_generator.tir.For)
    mod = gsm_data_generator.IRModule.from_expr(gsm_data_generator.tir.PrimFunc([Ab], wrapped))

    with gsm_data_generator.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_depth": 8, "explicit_unroll": False}}
    ):
        ret = gsm_data_generator.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret[0], gsm_data_generator.tir.For)
        assert ret[0].kind == gsm_data_generator.tir.ForKind.UNROLLED
        assert isinstance(ret[1], gsm_data_generator.tir.For)
        assert ret[1].kind != gsm_data_generator.tir.ForKind.UNROLLED


def test_unroll_fake_loop():
    ib = gsm_data_generator.tir.ir_builder.create()
    dtype = "int32"
    n = te.size_var("n")
    Ab = gsm_data_generator.tir.decl_buffer((n,), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(0, 1, name="i") as i:
        Aptr[i * 2] = 3
        with ib.for_range(0, 10, name="j") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()

    mod = gsm_data_generator.IRModule.from_expr(gsm_data_generator.tir.PrimFunc([Ab], stmt))

    with gsm_data_generator.transform.PassContext(
        config={
            "tir.UnrollLoop": {"auto_max_depth": 8, "auto_max_extent": 1, "explicit_unroll": False}
        }
    ):
        ret = gsm_data_generator.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret[0], gsm_data_generator.tir.BufferStore)


def test_unroll_allocations():
    @gsm_data_generator.script.ir_module
    class before:
        @T.prim_func
        def main():
            for i in T.unroll(2):
                with T.decl_buffer([16], "float32") as buf:
                    buf[0] = 0.0

    @gsm_data_generator.script.ir_module
    class expected:
        @T.prim_func
        def main():
            with T.decl_buffer([16], "float32") as buf1:
                buf1[0] = 0.0
            with T.decl_buffer([16], "float32") as buf2:
                buf2[0] = 0.0

    after = gsm_data_generator.tir.transform.UnrollLoop()(before)

    gsm_data_generator.ir.assert_structural_equal(after, expected)


def test_unroll_local_access():
    @gsm_data_generator.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((64,), "float32")):
            for bx in T.thread_binding(4, thread="blockIdx.x"):
                for tx in T.thread_binding(4, thread="threadIdx.x"):
                    A_local_data = T.allocate([4], dtype="float32", scope="local")
                    A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                    for i in T.serial(4):
                        A_local[i] = T.float32(i)

    @gsm_data_generator.script.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((64,), "float32")):
            for bx in T.thread_binding(4, thread="blockIdx.x"):
                for tx in T.thread_binding(4, thread="threadIdx.x"):
                    A_local_data = T.allocate([4], dtype="float32", scope="local")
                    A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                    A_local[0] = T.float32(0)
                    A_local[1] = T.float32(1)
                    A_local[2] = T.float32(2)
                    A_local[3] = T.float32(3)

    with gsm_data_generator.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_depth": 0,
                "auto_max_extent": 1,
                "explicit_unroll": True,
                "unroll_local_access": True,
            }
        }
    ):
        after = gsm_data_generator.tir.transform.UnrollLoop()(Before)
        after = gsm_data_generator.tir.transform.Simplify()(after)

    gsm_data_generator.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    test_unroll_local_access()
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_allocations()
