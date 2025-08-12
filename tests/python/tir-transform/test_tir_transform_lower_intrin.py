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
from gsm_data_generator import te
import numpy as np


def lower_intrin(params, stmt):
    """wrapper to call transformation in stmt"""
    lower_expr = isinstance(stmt, gsm_data_generator.tir.PrimExpr)
    stmt = gsm_data_generator.tir.Evaluate(stmt) if lower_expr else stmt
    mod = gsm_data_generator.IRModule.from_expr(
        gsm_data_generator.tir.PrimFunc(params, stmt).with_attr("target", gsm_data_generator.target.Target("llvm"))
    )
    mod = gsm_data_generator.transform.Sequential([gsm_data_generator.tir.transform.Simplify(), gsm_data_generator.tir.transform.LowerIntrin()])(
        mod
    )
    func = mod["main"]
    stmt = func.body
    return stmt.value if lower_expr else stmt.body


def check_value(expr, vx, vy, data, fref):
    n = len(data)
    A = te.placeholder((n,), name="A", dtype=expr.dtype)
    B = te.placeholder((n,), name="B", dtype=expr.dtype)

    def make_binds(i):
        x = expr
        x = gsm_data_generator.tir.Let(vx, A[i], x)
        x = gsm_data_generator.tir.Let(vy, B[i], x)
        return x

    C = te.compute((n,), make_binds)
    f = gsm_data_generator.compile(te.create_prim_func([A, B, C]), "llvm")
    a = gsm_data_generator.nd.array(np.array([x for x, y in data], dtype=expr.dtype))
    b = gsm_data_generator.nd.array(np.array([y for x, y in data], dtype=expr.dtype))
    c = gsm_data_generator.nd.array(np.zeros(len(data), dtype=expr.dtype))
    f(a, b, c)
    cref = np.array([fref(x, y) for x, y in data])
    np.testing.assert_equal(c.numpy(), cref)


def get_ref_data():
    """Get reference data for every pairs"""
    import itertools

    x = range(-10, 10)
    y = list(range(-10, 10))
    y.remove(0)
    return list(itertools.product(x, y))


@gsm_data_generator.testing.requires_llvm
def test_lower_floordiv():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = te.var("x", dtype=dtype)
        y = te.var("y", dtype=dtype)
        zero = gsm_data_generator.tir.const(0, dtype)
        # no constraints
        res = lower_intrin([x, y], gsm_data_generator.te.floordiv(x, y))
        check_value(res, x, y, data, lambda a, b: a // b)
        # rhs >= 0
        res = lower_intrin([x, y], gsm_data_generator.tir.Select(y >= 0, gsm_data_generator.te.floordiv(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a // b if b > 0 else 0)
        # involves max
        res = lower_intrin(
            [x, y], gsm_data_generator.tir.Select(y >= 0, gsm_data_generator.te.max(gsm_data_generator.te.floordiv(x, y), zero), zero)
        )
        check_value(res, x, y, data, lambda a, b: max(a // b, 0) if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(
            [x, y], gsm_data_generator.tir.Select(gsm_data_generator.tir.all(y >= 0, x >= 0), gsm_data_generator.te.floordiv(x, y), zero)
        )
        check_value(res, x, y, data, lambda a, b: a // b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin([x, y], gsm_data_generator.te.floordiv(x, gsm_data_generator.tir.const(8, dtype=dtype)))
        check_value(res, x, y, [(a, b) for a, b in data if b == 8], lambda a, b: a // b)


@gsm_data_generator.testing.requires_llvm
def test_lower_floormod():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = te.var("x", dtype=dtype)
        y = te.var("y", dtype=dtype)
        zero = gsm_data_generator.tir.const(0, dtype)
        # no constraints
        res = lower_intrin([x, y], gsm_data_generator.te.floormod(x, y))
        check_value(res, x, y, data, lambda a, b: a % b)
        # rhs >= 0
        res = lower_intrin([x, y], gsm_data_generator.tir.Select(y >= 0, gsm_data_generator.te.floormod(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a % b if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(
            [x, y], gsm_data_generator.tir.Select(gsm_data_generator.tir.all(y >= 0, x >= 0), gsm_data_generator.te.floormod(x, y), zero)
        )
        check_value(res, x, y, data, lambda a, b: a % b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin([x, y], gsm_data_generator.te.floormod(x, gsm_data_generator.tir.const(8, dtype=dtype)))
        check_value(res, x, y, [(a, b) for a, b in data if b == 8], lambda a, b: a % b)


if __name__ == "__main__":
    test_lower_floordiv()
    test_lower_floormod()
