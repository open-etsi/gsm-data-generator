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
from gsm_data_generator import te


def test_double_buffer():
    dtype = "int64"
    n = 100
    m = 4
    tx = te.thread_axis("threadIdx.x")
    ib = gsm_data_generator.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    ib.scope_attr(tx, "thread_extent", 1)
    with ib.for_range(0, n) as i:
        B = ib.allocate("float32", m, name="B", scope="shared")
        with ib.new_scope():
            ib.scope_attr(B.asobject().data, "double_buffer_scope", 1)
            with ib.for_range(0, m) as j:
                B[j] = A[i * 4 + j]
        with ib.for_range(0, m) as j:
            C[j] = B[j] + 1

    stmt = ib.get()
    mod = gsm_data_generator.IRModule({"db": gsm_data_generator.tir.PrimFunc([A.asobject(), C.asobject()], stmt)})

    opt = gsm_data_generator.transform.Sequential(
        [gsm_data_generator.tir.transform.InjectDoubleBuffer(), gsm_data_generator.tir.transform.Simplify()]
    )

    with gsm_data_generator.transform.PassContext(config={"tir.InjectDoubleBuffer": {"split_loop": 2}}):
        mod = opt(mod)
    stmt = mod["db"].body

    assert isinstance(stmt.body, gsm_data_generator.tir.Allocate)
    assert list(stmt.body.extents) == [m * 2]

    f = gsm_data_generator.tir.transform.ThreadSync("shared")(mod)["db"]
    count = [0]

    def count_sync(op):
        if isinstance(op, gsm_data_generator.tir.Call) and op.op.same_as(gsm_data_generator.ir.Op.get("tir.tvm_storage_sync")):
            count[0] += 1

    gsm_data_generator.tir.stmt_functor.post_order_visit(f.body, count_sync)
    assert count[0] == 4


class TestDoubleBuffer(gsm_data_generator.testing.CompareBeforeAfter):
    transform = gsm_data_generator.ir.transform.Sequential(
        [
            gsm_data_generator.tir.transform.InjectDoubleBuffer(),
            gsm_data_generator.tir.transform.Simplify(),
        ]
    )

    def before(A: T.Buffer([16, 32], "float32"), B: T.Buffer(16, "float32")):
        for i in range(16):
            cache_data = T.allocate([32], "float32")
            cache = T.Buffer(32, "float32", data=cache_data)

            T.attr(cache_data, "double_buffer_scope", 1)

            for j in range(32):
                cache[j] = A[i, j]

            B[i] = 0.0
            for j in range(32):
                B[i] = B[i] + cache[j]

    def expected(A: T.Buffer((16, 32), "float32"), B: T.Buffer((16,), "float32")):
        cache_data = T.allocate([64], "float32", "global")
        cache = T.Buffer(64, data=cache_data)
        for j in range(32):
            cache[j] = A[0, j]

        B[0] = T.float32(0)
        for j in range(32):
            B[0] = B[0] + cache[j]

        for i_outer in range(15):
            T.attr(cache_data, "double_buffer_write", 1)
            for j in range(32):
                cache[(i_outer + 1) % 2 * 32 + j] = A[i_outer + 1, j]
            B[i_outer + 1] = T.float32(0)
            for j in range(32):
                B[i_outer + 1] = B[i_outer + 1] + cache[(i_outer + 1) % 2 * 32 + j]


class TestDoubleBufferWithDeclBuffer(gsm_data_generator.testing.CompareBeforeAfter):
    """Like TestDoubleBuffer, but with a declared buffer object"""

    transform = gsm_data_generator.ir.transform.Sequential(
        [
            gsm_data_generator.tir.transform.InjectDoubleBuffer(),
            gsm_data_generator.tir.transform.Simplify(),
        ]
    )

    def before(A: T.Buffer((16, 32), "float32"), B: T.Buffer(16, "float32")):
        for i in range(16):
            cache = T.decl_buffer(32, "float32")
            T.attr(cache.data, "double_buffer_scope", 1)

            for j in range(32):
                cache[j] = A[i, j]

            B[i] = 0.0
            for j in range(32):
                B[i] = B[i] + cache[j]

    def expected(A: T.Buffer((16, 32), "float32"), B: T.Buffer(16, "float32")):
        cache = T.decl_buffer(64, "float32")
        for j in range(32):
            cache[j] = A[0, j]

        B[0] = T.float32(0)
        for j in range(32):
            B[0] = B[0] + cache[j]

        for i_outer in range(15):
            T.attr(cache.data, "double_buffer_write", 1)
            for j in range(32):
                cache[(i_outer + 1) % 2 * 32 + j] = A[i_outer + 1, j]
            B[i_outer + 1] = T.float32(0)
            for j in range(32):
                B[i_outer + 1] = B[i_outer + 1] + cache[(i_outer + 1) % 2 * 32 + j]


if __name__ == "__main__":
    gsm_data_generator.testing.main()
