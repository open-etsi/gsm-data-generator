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
import gsmDataGen.testing
import sys
import pytest
from gsmDataGen import te
import numpy as np


def test_const_saveload_json():
    # save load json
    x = gsmDataGen.tir.const(1, "int32")
    y = gsmDataGen.tir.const(10, "int32")
    z = x + y
    z = z + z
    json_str = gsmDataGen.ir.save_json(z)
    zz = gsmDataGen.ir.load_json(json_str)
    gsmDataGen.ir.assert_structural_equal(zz, z, map_free_vars=True)


def _test_infinity_value(value, dtype):
    x = gsmDataGen.tir.const(value, dtype)
    json_str = gsmDataGen.ir.save_json(x)
    gsmDataGen.ir.assert_structural_equal(x, gsmDataGen.ir.load_json(json_str))


def test_infinity_value():
    _test_infinity_value(float("inf"), "float64")
    _test_infinity_value(float("-inf"), "float64")
    _test_infinity_value(float("inf"), "float32")
    _test_infinity_value(float("-inf"), "float32")


def _test_minmax_value(value):
    json_str = gsmDataGen.ir.save_json(value)
    gsmDataGen.ir.assert_structural_equal(value, gsmDataGen.ir.load_json(json_str))


def test_minmax_value():
    _test_minmax_value(gsmDataGen.tir.min_value("float32"))
    _test_minmax_value(gsmDataGen.tir.max_value("float32"))


def test_make_smap():
    # save load json
    x = gsmDataGen.tir.const(1, "int32")
    y = gsmDataGen.tir.const(10, "int32")
    z = gsmDataGen.tir.Add(x, y)
    smap = gsmDataGen.runtime.convert({"z": z, "x": x})
    json_str = gsmDataGen.ir.save_json(gsmDataGen.runtime.convert([smap]))
    arr = gsmDataGen.ir.load_json(json_str)
    assert len(arr) == 1
    assert arr[0]["z"].a == arr[0]["x"]
    gsmDataGen.ir.assert_structural_equal(arr, [smap], map_free_vars=True)


def test_make_node():
    x = gsmDataGen.ir.make_node("ir.IntImm", dtype="int32", value=10, span=None)
    assert isinstance(x, gsmDataGen.tir.IntImm)
    assert x.value == 10
    A = te.placeholder((10,), name="A")
    AA = gsmDataGen.ir.make_node(
        "te.Tensor", shape=A.shape, dtype=A.dtype, op=A.op, value_index=A.value_index
    )
    assert AA.op == A.op
    assert AA.value_index == A.value_index

    y = gsmDataGen.ir.make_node("ir.IntImm", dtype=gsmDataGen.runtime.String("int32"), value=10, span=None)


def test_make_sum():
    A = te.placeholder((2, 10), name="A")
    k = te.reduce_axis((0, 10), "k")
    B = te.compute((2,), lambda i: te.sum(A[i, k], axis=k), name="B")
    json_str = gsmDataGen.ir.save_json(B)
    BB = gsmDataGen.ir.load_json(json_str)
    assert B.op.body[0].combiner is not None
    assert BB.op.body[0].combiner is not None


def test_env_func():
    @gsmDataGen.register_func("test.env_func")
    def test(x):
        return x + 1

    f = gsmDataGen.get_global_func("test.env_func")
    x = gsmDataGen.ir.EnvFunc.get("test.env_func")
    assert x.name == "test.env_func"
    json_str = gsmDataGen.ir.save_json([x])
    y = gsmDataGen.ir.load_json(json_str)[0]
    assert y.name == x.name
    assert y(1) == 2
    assert y.func(1) == 2

    x = gsmDataGen.ir.make_node("attrs.TestAttrs", name="xx", padding=(3, 4), func=y)
    assert x.name == "xx"
    assert x.padding[0].value == 3
    assert x.padding[1].value == 4
    assert x.axis == 10
    x = gsmDataGen.ir.load_json(gsmDataGen.ir.save_json(x))
    assert isinstance(x.func, gsmDataGen.ir.EnvFunc)
    assert x.func(10) == 11


def test_string():
    # non printable str, need to store by b64
    s1 = gsmDataGen.runtime.String("xy\x01z")
    s2 = gsmDataGen.ir.load_json(gsmDataGen.ir.save_json(s1))
    gsmDataGen.ir.assert_structural_equal(s1, s2)

    # printable str, need to store by repr_str
    s1 = gsmDataGen.runtime.String("xyz")
    s2 = gsmDataGen.ir.load_json(gsmDataGen.ir.save_json(s1))
    gsmDataGen.ir.assert_structural_equal(s1, s2)


def test_pass_config():
    cfg = gsmDataGen.transform.PassContext(
        opt_level=1,
        config={
            "tir.UnrollLoop": {
                "auto_max_step": 10,
            }
        },
    )
    cfg.opt_level == 1

    assert cfg.config["tir.UnrollLoop"].auto_max_step == 10
    # default option
    assert cfg.config["tir.UnrollLoop"].explicit_unroll == True

    # schema checking for specific config key
    with pytest.raises(TypeError):
        cfg = gsmDataGen.transform.PassContext(config={"tir.UnrollLoop": {"invalid": 1}})

    # schema check for un-registered config
    with pytest.raises(AttributeError):
        cfg = gsmDataGen.transform.PassContext(config={"inavlid-opt": True})

    # schema check for wrong type
    with pytest.raises(AttributeError):
        cfg = gsmDataGen.transform.PassContext(config={"tir.UnrollLoop": 1})


def test_dict():
    x = gsmDataGen.tir.const(1)  # a class that has Python-defined methods
    # instances should see the full class dict
    assert set(dir(x.__class__)) <= set(dir(x))


def test_ndarray():
    dev = gsmDataGen.cpu(0)
    tvm_arr = gsmDataGen.nd.array(np.random.rand(4), device=dev)
    tvm_arr2 = gsmDataGen.ir.load_json(gsmDataGen.ir.save_json(tvm_arr))
    gsmDataGen.ir.assert_structural_equal(tvm_arr, tvm_arr2)
    np.testing.assert_array_equal(tvm_arr.numpy(), tvm_arr2.numpy())


def test_ndarray_dict():
    dev = gsmDataGen.cpu(0)
    m1 = {
        "key1": gsmDataGen.nd.array(np.random.rand(4), device=dev),
        "key2": gsmDataGen.nd.array(np.random.rand(4), device=dev),
    }
    m2 = gsmDataGen.ir.load_json(gsmDataGen.ir.save_json(m1))
    gsmDataGen.ir.assert_structural_equal(m1, m2)


def test_free_var_equal():
    x = gsmDataGen.tir.Var("x", dtype="int32")
    y = gsmDataGen.tir.Var("y", dtype="int32")
    z = gsmDataGen.tir.Var("z", dtype="int32")
    v1 = x + y
    v1 = y + z
    gsmDataGen.ir.assert_structural_equal(x, z, map_free_vars=True)


def test_alloc_const():
    dev = gsmDataGen.cpu(0)
    dtype = "float32"
    shape = (16,)
    buf = gsmDataGen.tir.decl_buffer(shape, dtype)
    np_data = np.random.rand(*shape).astype(dtype)
    data = gsmDataGen.nd.array(np_data, device=dev)
    body = gsmDataGen.tir.Evaluate(0)
    alloc_const = gsmDataGen.tir.AllocateConst(buf.data, dtype, shape, data, body)
    alloc_const2 = gsmDataGen.ir.load_json(gsmDataGen.ir.save_json(alloc_const))
    gsmDataGen.ir.assert_structural_equal(alloc_const, alloc_const2)
    np.testing.assert_array_equal(np_data, alloc_const2.data.numpy())


if __name__ == "__main__":
    gsmDataGen.testing.main()
