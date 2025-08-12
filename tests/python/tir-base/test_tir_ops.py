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

import pytest


def check_throws(f):
    try:
        f()
    except gsm_data_generator.error.TVMError:
        pass
    else:
        raise AssertionError("Should have raised an exception but didn't.")


def test_const_fold():
    def check(f, *args):
        x = f(*[gsm_data_generator.tir.const(x, "int32") for x in args])
        y = f(*args)
        if not isinstance(x, (gsm_data_generator.tir.IntImm,)) or x.value != int(y):
            raise ValueError("check error: %s vs %s " % (x, y))

    tmod = gsm_data_generator.tir.truncmod
    check(lambda x, y: x + y, 3, 4)
    check(lambda x, y: x * y, 3, 12)
    check(lambda x, y: x * y - 10, 3, 12)
    check(lambda x, y: x - tmod(y, 10), 3, 12)
    check(lambda x, y: x // y + 10, 100, 12)
    check(lambda x, y: x & y + 10, 112, 128)
    check(lambda x, y: x > y, 112, 128)
    check(lambda x, y: x < y, 112, 128)
    check(lambda x, y: x <= y, 112, 128)
    check(lambda x, y: x >= y, 112, 128)
    check(lambda x, y: (x | y) ^ 10, 112, 128)


def test_const_fold2():
    x = te.var("x")
    tmod = gsm_data_generator.tir.truncmod
    tdiv = gsm_data_generator.tir.truncdiv
    assert (x + 0).same_as(x)
    assert (0 + x).same_as(x)
    assert (x - 0).same_as(x)
    assert tmod(x, 1).value == 0
    assert (x * 1).same_as(x)
    assert (1 * x).same_as(x)
    assert isinstance(tdiv(1, x), gsm_data_generator.tir.Div)


def test_const_fold3():
    # Test that using ints with logic operations is forbidden
    x = te.var("x")
    for val in [0, 1]:
        for func in [gsm_data_generator.tir.all, gsm_data_generator.tir.any]:
            check_throws(lambda: func(gsm_data_generator.tir.const(val, "uint1"), x))
            check_throws(lambda: func(x, gsm_data_generator.tir.const(val, "uint1")))

    # Test const folding when both arguments are const
    for tvm_func, py_func in [
        (gsm_data_generator.tir.all, lambda a, b: a and b),
        (gsm_data_generator.tir.any, lambda a, b: a or b),
    ]:
        for v1 in [0, 1]:
            for v2 in [0, 1]:
                gsm_data_generator.ir.assert_structural_equal(
                    tvm_func(gsm_data_generator.tir.const(v1, "uint1"), gsm_data_generator.tir.const(v2, "uint1")),
                    gsm_data_generator.tir.const(py_func(v1, v2), "uint1"),
                )

    x = te.var("x", "uint1")
    true = gsm_data_generator.tir.const(1, "uint1")
    false = gsm_data_generator.tir.const(0, "uint1")

    assert gsm_data_generator.tir.all(x, true).same_as(x)
    assert gsm_data_generator.tir.all(true, x).same_as(x)
    assert gsm_data_generator.tir.any(x, false).same_as(x)
    assert gsm_data_generator.tir.any(false, x).same_as(x)

    assert gsm_data_generator.tir.all(x, false).same_as(false)
    assert gsm_data_generator.tir.all(false, x).same_as(false)
    assert gsm_data_generator.tir.any(x, true).same_as(true)
    assert gsm_data_generator.tir.any(true, x).same_as(true)


def test_const_fold4():
    x1 = gsm_data_generator.tir.const(4, "int32")
    x2 = x1 + 5
    tdiv = gsm_data_generator.tir.truncdiv
    assert isinstance(x2, gsm_data_generator.tir.IntImm) and x2.value == 9
    x3 = tdiv(x2, 3)
    assert isinstance(x3, gsm_data_generator.tir.IntImm) and x3.value == 3
    x4 = x3 + 0.55
    assert isinstance(x4, gsm_data_generator.tir.FloatImm) and abs(x4.value - 3.55) < 1e-6
    x5 = te.ceil(x4)
    assert isinstance(x5, gsm_data_generator.tir.FloatImm) and x5.value == 4
    x6 = x5.astype("int")
    assert isinstance(x6, gsm_data_generator.tir.IntImm) and x6.value == 4, "x6={}".format(x6)
    y = (te.round((gsm_data_generator.tir.const(6.5, "float32") - 1) / 1.5) + 2).astype("int")
    assert isinstance(y, gsm_data_generator.tir.IntImm) and y.value == 6


def test_binary_dtype_match():
    def verify_general_dtype_support(f, is_conditional=False):
        rules = [
            [("bool", "int32"), "int32"],
            [("int32", "float32"), "float32"],
            [("int32", "int64"), "int64"],
            [("uint32", "int8"), "uint32"],
            [("uint32", "int32"), "uint32"],
        ]
        for (lhs_dtype, rhs_dtype), out_dtype in rules:
            lhs = te.var("lhs", dtype=lhs_dtype)
            rhs = te.var("rhs", dtype=rhs_dtype)
            out = f(lhs, rhs)
            if not is_conditional:
                assert out.dtype == out_dtype
            else:
                assert out.dtype == "bool"
            if hasattr(out, "a"):
                assert out.a.dtype == out_dtype
                assert out.b.dtype == out_dtype
            elif hasattr(out, "args"):
                # CallOp
                assert out.args[0].dtype == out_dtype
                assert out.args[1].dtype == out_dtype
            else:
                raise ValueError("Unknown binary op format!")

    def verify_callop_float_only(f):
        for lhs_dtype in ["int32", "float32", "float64"]:
            for rhs_dtype in ["int32", "float32", "float64"]:
                lhs = te.var("lhs", dtype=lhs_dtype)
                rhs = te.var("rhs", dtype=rhs_dtype)
                if "float" not in lhs_dtype and "float" not in rhs_dtype:
                    check_throws(lambda: f(lhs, rhs))
                elif "float" in lhs_dtype:
                    out = f(lhs, rhs)

                    # Upcasting for floating point types
                    dtypes = [lhs_dtype, rhs_dtype]
                    if "float64" in dtypes:
                        target_dtype = "float64"
                    elif "float32" in dtypes:
                        target_dtype = "float32"
                    else:
                        target_dtype = "int32"
                    assert out.dtype == target_dtype

                    # Final inputs are the right type
                    assert out.args[0].dtype == target_dtype
                    assert out.args[1].dtype == target_dtype
                else:
                    out = f(lhs, rhs)
                    assert out.dtype == rhs_dtype
                    assert out.args[0].dtype == rhs_dtype
                    assert out.args[1].dtype == rhs_dtype

    verify_general_dtype_support(lambda a, b: a + b)
    verify_general_dtype_support(lambda a, b: a * b)
    verify_general_dtype_support(lambda a, b: a >= b, is_conditional=True)
    verify_general_dtype_support(lambda a, b: a <= b, is_conditional=True)
    verify_callop_float_only(lambda a, b: te.power(a, b))

    # verify bool & int32 constant folding
    assert gsm_data_generator.tir.const(1) == gsm_data_generator.tir.const(True)
    assert gsm_data_generator.tir.const(2) != gsm_data_generator.tir.const(True)


def test_if_then_else():
    cases = [
        [(te.var("cond", dtype="bool"), "bool", "int32"), "int32"],
        [(True, "int32", "float32"), "float32"],
        [(False, "int32", "int64"), "int64"],
        [(te.var("cond", dtype="bool"), "uint32", "int32"), "uint32"],
        [(te.var("cond", dtype="int32"), "uint32", "int32"), "uint32"],
    ]
    for (cond, lhs_dtype, rhs_dtype), out_dtype in cases:
        lhs = te.var("lhs", dtype=lhs_dtype)
        rhs = te.var("rhs", dtype=rhs_dtype)
        if cond is True or cond is False:
            out = gsm_data_generator.tir.if_then_else(cond, lhs, rhs)
            out2 = gsm_data_generator.tir.if_then_else(not cond, rhs, lhs)
            out3 = gsm_data_generator.tir.if_then_else(not cond, lhs, rhs)
            gsm_data_generator.ir.assert_structural_equal(out, out2) == 1
            if cond:
                gsm_data_generator.ir.assert_structural_equal(out, lhs.astype(out_dtype)) == 1
                gsm_data_generator.ir.assert_structural_equal(out3, rhs.astype(out_dtype)) == 1
            else:
                gsm_data_generator.ir.assert_structural_equal(out, rhs.astype(out_dtype)) == 1
                gsm_data_generator.ir.assert_structural_equal(out3, lhs.astype(out_dtype)) == 1
        elif cond.dtype == "bool":
            out = gsm_data_generator.tir.if_then_else(cond, lhs, rhs)
            assert out.dtype == out_dtype
            assert out.args[1].dtype == out_dtype
            assert out.args[2].dtype == out_dtype
        elif cond.dtype != "bool":
            check_throws(lambda: gsm_data_generator.tir.if_then_else(cond, lhs, rhs))
        else:
            raise ValueError("Unknown combinations")


@pytest.mark.parametrize("num_args", list(range(2, 10)))
def test_comm_reducer(num_args):
    """Handle all arguments in tir comm_reducer

    The `tir.comm_reducer` API has two distinct usages.  It can reduce
    a tensor along a specified axis, similar to numpy.max, or it can
    reduce several arguments together, simililar to Python's built-in
    max().  This choice is based on the type of the second argument.

    If the `tir.comm_reducer` is reducing all arguments, then all
    arguments should be used.  In the past, the introduction of new
    arguments intended for use when reducing along a tensor axis has
    failed to forward these arguments when reducing along a list of
    items.
    """
    assert gsm_data_generator.tir.max(*range(num_args)) == num_args - 1


def test_llvm_intrin():
    with pytest.raises(ValueError, match=r"Unknown llvm intrinsic function llvm.dummy"):
        a = gsm_data_generator.tir.call_llvm_intrin("int32x4", "llvm.dummy", 0)
    with pytest.raises(ValueError, match=r"Unknown llvm intrinsic function llvm.dummy"):
        a = gsm_data_generator.tir.call_llvm_pure_intrin("int32x4", "llvm.dummy", 0)


if __name__ == "__main__":
    gsm_data_generator.testing.main()
