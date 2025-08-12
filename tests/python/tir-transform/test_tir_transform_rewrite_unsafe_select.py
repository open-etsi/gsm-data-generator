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


def test_rewrite_Select():
    ib = gsm_data_generator.tir.ir_builder.create()
    A = ib.allocate("float32", 100, name="A", scope="global")
    i = te.var("i")
    y = gsm_data_generator.tir.Select(i > 1, A[i - 1], 1.0)

    mod = gsm_data_generator.IRModule.from_expr(gsm_data_generator.tir.PrimFunc([i], gsm_data_generator.tir.Evaluate(y)))
    yy = gsm_data_generator.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value

    z = gsm_data_generator.tir.Select(gsm_data_generator.tir.Select(i > 1, A[i - 1], 1.0) > 0.0, A[i], 0.1)
    mod = gsm_data_generator.IRModule.from_expr(gsm_data_generator.tir.PrimFunc([i], gsm_data_generator.tir.Evaluate(z)))
    zz = gsm_data_generator.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value

    a = gsm_data_generator.tir.Select(gsm_data_generator.tir.floordiv(i, 4) > 10, y, z)

    mod = gsm_data_generator.IRModule.from_expr(gsm_data_generator.tir.PrimFunc([i], gsm_data_generator.tir.Evaluate(a)))
    aa = gsm_data_generator.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value
    builtin_if_then_else = gsm_data_generator.ir.Op.get("tir.if_then_else")

    assert yy.op.same_as(builtin_if_then_else)
    assert yy.op.same_as(builtin_if_then_else)
    assert isinstance(aa, gsm_data_generator.tir.Select)


if __name__ == "__main__":
    test_rewrite_Select()
