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
import sys
import pytest
from gsm_data_generator import te
import numpy as np


def test_const_saveload_json():
    # save load json
    x = gsm_data_generator.tir.const(1, "int32")
    y = gsm_data_generator.tir.const(10, "int32")
    z = x + y
    z = z + z
    json_str = gsm_data_generator.ir.save_json(z)
    zz = gsm_data_generator.ir.load_json(json_str)
    gsm_data_generator.ir.assert_structural_equal(zz, z, map_free_vars=True)


def _test_infinity_value(value, dtype):
    x = gsm_data_generator.tir.const(value, dtype)
    json_str = gsm_data_generator.ir.save_json(x)
    gsm_data_generator.ir.assert_structural_equal(x, gsm_data_generator.ir.load_json(json_str))


def test_infinity_value():
    _test_infinity_value(float("inf"), "float64")
    _test_infinity_value(float("-inf"), "float64")
    _test_infinity_value(float("inf"), "float32")
    _test_infinity_value(float("-inf"), "float32")


def _test_minmax_value(value):
    json_str = gsm_data_generator.ir.save_json(value)
    gsm_data_generator.ir.assert_structural_equal(value, gsm_data_generator.ir.load_json(json_str))


def test_minmax_value():
    _test_minmax_value(gsm_data_generator.tir.min_value("float32"))
    _test_minmax_value(gsm_data_generator.tir.max_value("float32"))


def test_make_smap():
    # save load json
    x = gsm_data_generator.tir.const(1, "int32")
    y = gsm_data_generator.tir.const(10, "int32")
    z = gsm_data_generator.tir.Add(x, y)
    smap = gsm_data_generator.runtime.convert({"z": z, "x": x})
    json_str = gsm_data_generator.ir.save_json(gsm_data_generator.runtime.convert([smap]))
    arr = gsm_data_generator.ir.load_json(json_str)
    assert len(arr) == 1
    assert arr[0]["z"].a == arr[0]["x"]
    gsm_data_generator.ir.assert_structural_equal(arr, [smap], map_free_vars=True)


def test_make_node():
    x = gsm_data_generator.ir.make_node("ir.IntImm", dtype="int32", value=10, span=None)
    assert isinstance(x, gsm_data_generator.tir.IntImm)
    assert x.value == 10
    A = te.placeholder((10,), name="A")
    AA = gsm_data_generator.ir.make_node(
        "te.Tensor", shape=A.shape, dtype=A.dtype, op=A.op, value_index=A.value_index
    )
    assert AA.op == A.op
    assert AA.value_index == A.value_index

    y = gsm_data_generator.ir.make_node("ir.IntImm", dtype=gsm_data_generator.runtime.String("int32"), value=10, span=None)


def test_make_sum():
    A = te.placeholder((2, 10), name="A")
    k = te.reduce_axis((0, 10), "k")
    B = te.compute((2,), lambda i: te.sum(A[i, k], axis=k), name="B")
    json_str = gsm_data_generator.ir.save_json(B)
    BB = gsm_data_generator.ir.load_json(json_str)
    assert B.op.body[0].combiner is not None
    assert BB.op.body[0].combiner is not None


def test_env_func():
    @gsm_data_generator.register_func("test.env_func")
    def test(x):
        return x + 1

    f = gsm_data_generator.get_global_func("test.env_func")
    x = gsm_data_generator.ir.EnvFunc.get("test.env_func")
    assert x.name == "test.env_func"
    json_str = gsm_data_generator.ir.save_json([x])
    y = gsm_data_generator.ir.load_json(json_str)[0]
    assert y.name == x.name
    assert y(1) == 2
    assert y.func(1) == 2

    x = gsm_data_generator.ir.make_node("attrs.TestAttrs", name="xx", padding=(3, 4), func=y)
    assert x.name == "xx"
    assert x.padding[0].value == 3
    assert x.padding[1].value == 4
    assert x.axis == 10
    x = gsm_data_generator.ir.load_json(gsm_data_generator.ir.save_json(x))
    assert isinstance(x.func, gsm_data_generator.ir.EnvFunc)
    assert x.func(10) == 11


def test_string():
    # non printable str, need to store by b64
    s1 = gsm_data_generator.runtime.String("xy\x01z")
    s2 = gsm_data_generator.ir.load_json(gsm_data_generator.ir.save_json(s1))
    gsm_data_generator.ir.assert_structural_equal(s1, s2)

    # printable str, need to store by repr_str
    s1 = gsm_data_generator.runtime.String("xyz")
    s2 = gsm_data_generator.ir.load_json(gsm_data_generator.ir.save_json(s1))
    gsm_data_generator.ir.assert_structural_equal(s1, s2)


def test_pass_config():
    cfg = gsm_data_generator.transform.PassContext(
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
        cfg = gsm_data_generator.transform.PassContext(config={"tir.UnrollLoop": {"invalid": 1}})

    # schema check for un-registered config
    with pytest.raises(AttributeError):
        cfg = gsm_data_generator.transform.PassContext(config={"inavlid-opt": True})

    # schema check for wrong type
    with pytest.raises(AttributeError):
        cfg = gsm_data_generator.transform.PassContext(config={"tir.UnrollLoop": 1})


def test_dict():
    x = gsm_data_generator.tir.const(1)  # a class that has Python-defined methods
    # instances should see the full class dict
    assert set(dir(x.__class__)) <= set(dir(x))


def test_ndarray():
    dev = gsm_data_generator.cpu(0)
    tvm_arr = gsm_data_generator.nd.array(np.random.rand(4), device=dev)
    tvm_arr2 = gsm_data_generator.ir.load_json(gsm_data_generator.ir.save_json(tvm_arr))
    gsm_data_generator.ir.assert_structural_equal(tvm_arr, tvm_arr2)
    np.testing.assert_array_equal(tvm_arr.numpy(), tvm_arr2.numpy())


def test_ndarray_dict():
    dev = gsm_data_generator.cpu(0)
    m1 = {
        "key1": gsm_data_generator.nd.array(np.random.rand(4), device=dev),
        "key2": gsm_data_generator.nd.array(np.random.rand(4), device=dev),
    }
    m2 = gsm_data_generator.ir.load_json(gsm_data_generator.ir.save_json(m1))
    gsm_data_generator.ir.assert_structural_equal(m1, m2)


def test_free_var_equal():
    x = gsm_data_generator.tir.Var("x", dtype="int32")
    y = gsm_data_generator.tir.Var("y", dtype="int32")
    z = gsm_data_generator.tir.Var("z", dtype="int32")
    v1 = x + y
    v1 = y + z
    gsm_data_generator.ir.assert_structural_equal(x, z, map_free_vars=True)


def test_alloc_const():
    dev = gsm_data_generator.cpu(0)
    dtype = "float32"
    shape = (16,)
    buf = gsm_data_generator.tir.decl_buffer(shape, dtype)
    np_data = np.random.rand(*shape).astype(dtype)
    data = gsm_data_generator.nd.array(np_data, device=dev)
    body = gsm_data_generator.tir.Evaluate(0)
    alloc_const = gsm_data_generator.tir.AllocateConst(buf.data, dtype, shape, data, body)
    alloc_const2 = gsm_data_generator.ir.load_json(gsm_data_generator.ir.save_json(alloc_const))
    gsm_data_generator.ir.assert_structural_equal(alloc_const, alloc_const2)
    np.testing.assert_array_equal(np_data, alloc_const2.data.numpy())


if __name__ == "__main__":
    gsm_data_generator.testing.main()
