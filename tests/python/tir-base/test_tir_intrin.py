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
from gsm_data_generator import te, tir
from gsm_data_generator import topi
from gsm_data_generator.contrib import utils, clang
from gsm_data_generator.script import tir as T
import numpy as np
import ctypes
import math
import scipy


def test_nearbyint():
    m = te.var(
        "m",
    )
    A = te.placeholder((m,), name="A")
    A_rounded = te.compute((m,), lambda *i: gsm_data_generator.tir.nearbyint(A(*i)), name="A")

    # Convert to TIR and create schedule
    mod = te.create_prim_func([A, A_rounded])
    sch = tir.Schedule(mod)

    # Build from scheduled TIR
    func = gsm_data_generator.compile(sch.mod, target="llvm")

    dev = gsm_data_generator.cpu(0)
    n = 10
    a = gsm_data_generator.nd.array(np.random.uniform(high=100, size=n).astype(A.dtype), dev)
    a_rounded = gsm_data_generator.nd.array(np.random.uniform(size=n).astype(A_rounded.dtype), dev)
    func(a, a_rounded)
    # Note that numpys rint rounds to nearest integer with
    # ties to halfway is broken by rounding to even.
    # So that 1.5 and 2.5 will round 2.
    # This is the default rounding mode with libc as well.
    # However one can set a different rounding mode and in that
    # case numpy result might differ.
    gsm_data_generator.testing.assert_allclose(a_rounded.numpy(), np.rint(a.numpy()))


def test_round_intrinsics_on_int():
    i = gsm_data_generator.te.var("i", "int32")
    for op in [gsm_data_generator.tir.round, gsm_data_generator.tir.trunc, gsm_data_generator.tir.ceil, gsm_data_generator.tir.floor, gsm_data_generator.tir.nearbyint]:
        assert op(gsm_data_generator.tir.const(10, "int32")).value == 10
        assert op(gsm_data_generator.tir.const(True, "bool")).value == True
        assert op(i).same_as(i)

    assert gsm_data_generator.tir.isnan(gsm_data_generator.tir.const(10, "int32")).value == False


def test_unary_intrin():
    test_funcs = [
        (gsm_data_generator.tir.exp10, lambda x: np.power(10, x)),
        (gsm_data_generator.tir.log2, lambda x: np.log2(x)),
        (gsm_data_generator.tir.log10, lambda x: np.log10(x)),
        (gsm_data_generator.tir.sinh, lambda x: np.sinh(x)),
        (gsm_data_generator.tir.cosh, lambda x: np.cosh(x)),
        (gsm_data_generator.tir.log1p, lambda x: np.log1p(x)),
        (gsm_data_generator.tir.asin, lambda x: np.arcsin(x)),
        (gsm_data_generator.tir.acos, lambda x: np.arccos(x)),
        (gsm_data_generator.tir.atan, lambda x: np.arctan(x)),
        (gsm_data_generator.tir.asinh, lambda x: np.arcsinh(x)),
        (gsm_data_generator.tir.acosh, lambda x: np.arccosh(x)),
        (gsm_data_generator.tir.atanh, lambda x: np.arctanh(x)),
        (gsm_data_generator.tir.erf, lambda x: scipy.special.erf(x)),
    ]

    def run_test(tvm_intrin, np_func, atol=1e-5, rtol=1e-5):
        m = te.var(
            "m",
        )
        A = te.placeholder((m,), name="A")
        B = te.compute((m,), lambda *i: tvm_intrin(A(*i)), name="B")

        # Convert to TIR and create schedule
        mod = te.create_prim_func([A, B])
        sch = tir.Schedule(mod)

        # Build from scheduled TIR
        func = gsm_data_generator.compile(sch.mod, target="llvm")

        dev = gsm_data_generator.cpu(0)
        n = 10
        a = gsm_data_generator.nd.array(np.random.uniform(0.1, 0.5, size=n).astype(A.dtype), dev)
        b = gsm_data_generator.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        func(a, b)
        gsm_data_generator.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=atol, rtol=rtol)

        # Out‐of‐bounds test for asin/acos
        name = tvm_intrin.__name__
        if name in ("asin", "acos"):
            # generate some values outside [-1, 1]
            n = 8
            out_np = np.concatenate(
                [
                    np.random.uniform(1.1, 2.0, size=n // 2),
                    np.random.uniform(-2.0, -1.1, size=n // 2),
                ]
            ).astype(A.dtype)
            a2 = gsm_data_generator.nd.array(out_np, dev)
            b2 = gsm_data_generator.nd.array(np.empty_like(out_np), dev)
            func(a2, b2)
            # all outputs should be NaN
            assert np.all(np.isnan(b2.numpy()))

    for func in test_funcs:
        atol = rtol = 1e-3 if func[0].__name__ in ["asin", "acos", "atan"] else 1e-5
        run_test(*func, atol, rtol)


def test_binary_intrin():
    test_funcs = [
        (gsm_data_generator.tir.atan2, lambda x1, x2: np.arctan2(x1, x2)),
        (gsm_data_generator.tir.nextafter, lambda x1, x2: np.nextafter(x1, x2)),
        (gsm_data_generator.tir.copysign, lambda x1, x2: np.copysign(x1, x2)),
        (gsm_data_generator.tir.hypot, lambda x1, x2: np.hypot(x1, x2)),
    ]

    def run_test(tvm_intrin, np_func):
        m = te.var(
            "m",
        )
        A = te.placeholder((m,), name="A")
        B = te.placeholder((m,), name="B")
        C = te.compute((m,), lambda *i: tvm_intrin(A(*i), B(*i)), name="C")

        # Convert to TIR and create schedule
        mod = te.create_prim_func([A, B, C])
        sch = tir.Schedule(mod)

        # Build from scheduled TIR
        func = gsm_data_generator.compile(sch.mod, target="llvm")

        dev = gsm_data_generator.cpu(0)
        n = 10
        a = gsm_data_generator.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), dev)
        b = gsm_data_generator.nd.array(np.random.uniform(0, 1, size=n).astype(B.dtype), dev)
        c = gsm_data_generator.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        func(a, b, c)
        gsm_data_generator.testing.assert_allclose(c.numpy(), np_func(a.numpy(), b.numpy()), atol=1e-5, rtol=1e-5)

    for func in test_funcs:
        run_test(*func)


def test_ldexp():
    m = te.var(
        "m",
    )
    A = te.placeholder((m,), name="A")
    B = te.placeholder((m,), name="B", dtype="int32")
    C = te.compute((m,), lambda *i: gsm_data_generator.tir.ldexp(A(*i), B(*i)), name="C")

    # Convert to TIR and create schedule
    mod = te.create_prim_func([A, B, C])
    sch = tir.Schedule(mod)

    # Build from scheduled TIR
    func = gsm_data_generator.compile(sch.mod, target="llvm")

    dev = gsm_data_generator.cpu(0)
    n = 10
    a = gsm_data_generator.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), dev)
    b = gsm_data_generator.nd.array(np.random.randint(0, 5, size=n).astype(B.dtype), dev)
    c = gsm_data_generator.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    func(a, b, c)
    gsm_data_generator.testing.assert_allclose(c.numpy(), np.ldexp(a.numpy(), b.numpy()), atol=1e-5, rtol=1e-5)


dtype = gsm_data_generator.testing.parameter("int32", "int64")


@gsm_data_generator.testing.parametrize_targets("llvm", "vulkan -from_device=0")
def test_clz(target, dev, dtype):
    target = gsm_data_generator.target.Target(target)
    if (
        target.kind.name == "vulkan"
        and dtype == "int64"
        and not target.attrs.get("supports_int64", False)
    ):
        pytest.xfail("Vulkan target does not support Int64 types")

    def clz_np(x, dtype):
        ceil_log2 = np.ceil(np.log2(x)).astype(dtype)
        bits = int(dtype[-2:])
        clz = bits - ceil_log2
        clz[np.bitwise_and(x, x - 1) == 0] -= 1
        return clz

    m = te.var("m")
    A = te.placeholder((m,), name="A", dtype=dtype)
    B = te.compute((m,), lambda *i: gsm_data_generator.tir.clz(A(*i)), name="B")

    # Convert to TIR and create schedule
    mod = te.create_prim_func([A, B])
    sch = tir.Schedule(mod)

    # Apply scheduling primitives if target is Vulkan
    if target.kind.name == "vulkan":
        block = sch.get_block("B")
        loop = sch.get_loops(block)[0]
        bx, tx = sch.split(loop, factors=[None, 64])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

    # Build from scheduled TIR
    func = gsm_data_generator.compile(sch.mod, target=target)

    n = 10
    highs = [10, 100, 1000, 10000, 100000, 1000000]

    if dtype == "int64":
        highs.append((1 << 63) - 1)

    for high in highs:
        a_np = np.random.randint(1, high=high, size=(n,), dtype=dtype)
        a = gsm_data_generator.nd.array(a_np, dev)
        b = gsm_data_generator.nd.array(np.zeros((n,)).astype("int32"), dev)
        func(a, b)
        ref = clz_np(a_np, dtype)
        np.testing.assert_equal(b.numpy(), ref)


@gsm_data_generator.script.ir_module
class Module:
    @T.prim_func
    def test_tir_fma(A: T.handle, B: T.handle, C: T.handle, d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_fma", "tir.noalias": True})
        n = T.int32()
        stride = T.int32()
        stride_1 = T.int32()
        stride_2 = T.int32()
        stride_3 = T.int32()
        A_1 = T.match_buffer(
            A,
            [n],
            strides=[stride],
            elem_offset=0,
            align=64,
            offset_factor=1,
            buffer_type="auto",
        )
        B_1 = T.match_buffer(
            B,
            [n],
            strides=[stride_1],
            elem_offset=0,
            align=64,
            offset_factor=1,
            buffer_type="auto",
        )
        C_1 = T.match_buffer(
            C,
            [n],
            strides=[stride_2],
            elem_offset=0,
            align=64,
            offset_factor=1,
            buffer_type="auto",
        )
        d_1 = T.match_buffer(
            d,
            [n],
            strides=[stride_3],
            elem_offset=0,
            align=64,
            offset_factor=1,
            buffer_type="auto",
        )
        # body
        for i in T.serial(0, n):
            d_1[(i * stride_3)] = (A_1[(i * stride)] * B_1[(i * stride_1)]) + C_1[(i * stride_2)]


def test_fma():
    opt = gsm_data_generator.transform.Sequential(
        [
            gsm_data_generator.tir.transform.Apply(lambda f: f.with_attr("target", gsm_data_generator.target.Target("llvm"))),
            gsm_data_generator.tir.transform.LowerIntrin(),
        ]
    )
    mod = opt(Module)
    assert mod["test_tir_fma"].body.body.value.op.name == "tir.call_llvm_pure_intrin"


if __name__ == "__main__":
    test_nearbyint()
    test_unary_intrin()
    test_round_intrinsics_on_int()
    test_binary_intrin()
    test_ldexp()
    test_clz()
    test_fma()
