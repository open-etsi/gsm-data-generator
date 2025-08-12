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
import gsmDataGen
from gsmDataGen import te
from gsmDataGen.script import tir as T
import os


def test_unroll_loop():
    ib = gsmDataGen.tir.ir_builder.create()
    dtype = "int64"
    n = te.size_var("n")
    Ab = gsmDataGen.tir.decl_buffer((n,), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(n, n + 2, name="i") as i:
        with ib.for_range(0, 8, name="i", kind="unroll") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()
    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([Ab], stmt))

    assert isinstance(stmt, gsmDataGen.tir.For)

    with gsmDataGen.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        ret = gsmDataGen.tir.transform.UnrollLoop()(mod)["main"].body
        assert not isinstance(ret, gsmDataGen.tir.For)

    with gsmDataGen.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 15}}):
        ret = gsmDataGen.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret, gsmDataGen.tir.For)

    with gsmDataGen.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_step": 16, "explicit_unroll": False}}
    ):
        ret = gsmDataGen.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret, gsmDataGen.tir.For)
        assert ret.kind == gsmDataGen.tir.ForKind.UNROLLED

    ib = gsmDataGen.tir.ir_builder.create()
    ib.scope_attr(gsmDataGen.tir.const(0, "int32"), "pragma_auto_unroll_max_step", 16)
    ib.emit(stmt)
    wrapped = ib.get()
    wrapped = gsmDataGen.tir.SeqStmt([wrapped, stmt])
    assert isinstance(ret, gsmDataGen.tir.For)
    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([Ab], wrapped))

    with gsmDataGen.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_depth": 8, "explicit_unroll": False}}
    ):
        ret = gsmDataGen.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret[0], gsmDataGen.tir.For)
        assert ret[0].kind == gsmDataGen.tir.ForKind.UNROLLED
        assert isinstance(ret[1], gsmDataGen.tir.For)
        assert ret[1].kind != gsmDataGen.tir.ForKind.UNROLLED


def test_unroll_fake_loop():
    ib = gsmDataGen.tir.ir_builder.create()
    dtype = "int32"
    n = te.size_var("n")
    Ab = gsmDataGen.tir.decl_buffer((n,), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(0, 1, name="i") as i:
        Aptr[i * 2] = 3
        with ib.for_range(0, 10, name="j") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()

    mod = gsmDataGen.IRModule.from_expr(gsmDataGen.tir.PrimFunc([Ab], stmt))

    with gsmDataGen.transform.PassContext(
        config={
            "tir.UnrollLoop": {"auto_max_depth": 8, "auto_max_extent": 1, "explicit_unroll": False}
        }
    ):
        ret = gsmDataGen.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret[0], gsmDataGen.tir.BufferStore)


def test_unroll_allocations():
    @gsmDataGen.script.ir_module
    class before:
        @T.prim_func
        def main():
            for i in T.unroll(2):
                with T.decl_buffer([16], "float32") as buf:
                    buf[0] = 0.0

    @gsmDataGen.script.ir_module
    class expected:
        @T.prim_func
        def main():
            with T.decl_buffer([16], "float32") as buf1:
                buf1[0] = 0.0
            with T.decl_buffer([16], "float32") as buf2:
                buf2[0] = 0.0

    after = gsmDataGen.tir.transform.UnrollLoop()(before)

    gsmDataGen.ir.assert_structural_equal(after, expected)


def test_unroll_local_access():
    @gsmDataGen.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((64,), "float32")):
            for bx in T.thread_binding(4, thread="blockIdx.x"):
                for tx in T.thread_binding(4, thread="threadIdx.x"):
                    A_local_data = T.allocate([4], dtype="float32", scope="local")
                    A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                    for i in T.serial(4):
                        A_local[i] = T.float32(i)

    @gsmDataGen.script.ir_module
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

    with gsmDataGen.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_depth": 0,
                "auto_max_extent": 1,
                "explicit_unroll": True,
                "unroll_local_access": True,
            }
        }
    ):
        after = gsmDataGen.tir.transform.UnrollLoop()(Before)
        after = gsmDataGen.tir.transform.Simplify()(after)

    gsmDataGen.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    test_unroll_local_access()
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_allocations()
