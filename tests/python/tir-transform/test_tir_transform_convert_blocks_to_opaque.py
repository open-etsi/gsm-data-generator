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
from gsm_data_generator import tir, te
from gsm_data_generator.script import tir as T


def _check(original, transformed):
    func = original
    mod = gsm_data_generator.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = gsm_data_generator.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = gsm_data_generator.tir.transform.Simplify()(mod)
    gsm_data_generator.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"))


@T.prim_func
def elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer((16, 16), "float32")
            for j in range(0, 16):
                with T.block():
                    vi = T.axis.S(16, i)
                    vj = T.axis.S(16, j)
                    B[vi, vj] = A[vi, vj] + 1.0
            for j in range(0, 16):
                with T.block():
                    vi = T.axis.S(16, i)
                    vj = T.axis.S(16, j)
                    C[vi, vj] = B[vi, vj] * 2.0


@T.prim_func
def substituted_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer([16, 16], "float32")
            for j in range(0, 16):
                with T.block():
                    T.reads([A[i, j]])
                    T.writes([B[i, j]])
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block():
                    T.reads([B[i, j]])
                    T.writes([C[i, j]])
                    C[i, j] = B[i, j] * 2.0


def test_elementwise():
    _check(elementwise_func, substituted_elementwise_func)


class TestErrorIfPredicateUsesBlockVariables(gsm_data_generator.testing.CompareBeforeAfter):
    transform = gsm_data_generator.tir.transform.ConvertBlocksToOpaque()
    check_well_formed = False

    def before(A: T.Buffer(8, "int32")):
        for i in T.serial(8):
            with T.block():
                vi = T.axis.remap("S", [i])
                T.where(vi < 6)
                T.evaluate(0)

    expected = gsm_data_generator.TVMError


if __name__ == "__main__":
    gsm_data_generator.testing.main()
