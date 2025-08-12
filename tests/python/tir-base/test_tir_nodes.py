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
import numpy as np
import pytest
import gsm_data_generator
from gsm_data_generator import ir, te


def test_const():
    x = gsm_data_generator.tir.const(1, "int32")
    assert x.dtype == "int32"
    assert isinstance(x, gsm_data_generator.tir.IntImm)


def test_te_const():
    x = gsm_data_generator.te.const(1, "int32")
    assert x.dtype == "int32"
    assert isinstance(x, gsm_data_generator.tir.IntImm)


def test_tir_const_dtype_inference():
    for data in [
        True,
        bool(1),
        np.uint8(1),
        np.uint16(1),
        np.uint32(1),
        np.uint64(1),
        np.int8(1),
        np.int16(1),
        np.int32(1),
        np.int64(1),
        np.float16(1),
        np.float32(1),
        np.float64(1),
    ]:
        assert gsm_data_generator.tir.const(data).dtype == str(np.array(data).dtype)

    assert gsm_data_generator.tir.const(True).dtype == "bool"
    assert gsm_data_generator.tir.const(1).dtype == "int32"
    assert gsm_data_generator.tir.const(1.0).dtype == "float32"


def test_make():
    x = gsm_data_generator.tir.const(1, "int32")
    y = te.var("x")
    z = x + y
    assert isinstance(gsm_data_generator.tir.max(x, y), gsm_data_generator.tir.Max)
    assert isinstance(gsm_data_generator.tir.min(x, y), gsm_data_generator.tir.Min)


def test_ir():
    x = gsm_data_generator.tir.const(1, "int32")
    y = gsm_data_generator.tir.IntImm("int32", 1)
    z = x + y
    stmt = gsm_data_generator.tir.Evaluate(z)
    assert isinstance(stmt, gsm_data_generator.tir.Evaluate)


def test_ir2():
    buf_size = te.var("size")
    x = te.var("n")

    storage_type = ir.PrimType("int32")
    handle_type = ir.PointerType(storage_type)
    array = te.var("array", handle_type)
    buf = gsm_data_generator.tir.decl_buffer([buf_size], "int32", data=array)

    st = gsm_data_generator.tir.BufferStore(buf, x + 1, [1])
    assert isinstance(st, gsm_data_generator.tir.BufferStore)
    assert st.buffer == buf
    assert st.buffer.data == array


def test_let():
    x = te.var("x")
    y = te.var("y")
    stmt = gsm_data_generator.tir.LetStmt(x, 10, gsm_data_generator.tir.Evaluate(x + 1))


def test_cast():
    x = te.var("x", dtype="float32")
    y = x.astype("int32")
    z = x.astype("float32x4")
    assert isinstance(y, gsm_data_generator.tir.Cast)
    assert isinstance(z, gsm_data_generator.tir.Broadcast)
    assert z.lanes == 4

    s = gsm_data_generator.tir.StringImm("s")
    with pytest.raises(gsm_data_generator.error.TVMError):
        try:
            s.astype("int")
        except Exception as e:
            assert "Can't cast a handle to other types" in str(e)
            raise


def test_attr():
    x = te.var("x")
    y = te.var("y")
    stmt = gsm_data_generator.tir.AttrStmt(y, "stride", 10, gsm_data_generator.tir.Evaluate(x + 1))
    assert stmt.node == y

    a = gsm_data_generator.runtime.convert(1)
    assert a == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass


def test_basic():
    a = te.var("a")
    b = te.var("b")
    c = a + b
    assert str(c) == "%s + %s" % (a.name, b.name)


def test_stmt():
    x = gsm_data_generator.tir.Evaluate(0)
    gsm_data_generator.tir.For(te.var("i"), 0, 1, gsm_data_generator.tir.ForKind.SERIAL, x)


def test_dir():
    x = te.var("x")
    dir(x)


def test_dtype():
    x = te.var("x")
    assert x.dtype == "int32"
    y = te.var("y")
    assert (x > y).dtype == "bool"


def test_any():
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    try:
        t = x or x
        assert False
    except ValueError:
        pass
    try:
        gsm_data_generator.tir.any()
        assert False
    except ValueError:
        pass
    assert str(gsm_data_generator.tir.any(x < y)) == "%s < %s" % (x.name, y.name)
    assert str(gsm_data_generator.tir.any(x < y, x > z)) == "%s < %s or %s > %s" % (
        x.name,
        y.name,
        x.name,
        z.name,
    )
    assert str(
        gsm_data_generator.tir.any(x < y, y > z + 1, x < z * 2)
    ) == "%s < %s or %s > %s + 1 or %s < %s * 2" % (
        x.name,
        y.name,
        y.name,
        z.name,
        x.name,
        z.name,
    )


def test_all():
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    try:
        t = x and x
        assert False
    except ValueError:
        pass
    try:
        gsm_data_generator.tir.all()
        assert False
    except ValueError:
        pass
    assert str(gsm_data_generator.tir.all(x < y)) == "%s < %s" % (x.name, y.name)
    assert str(gsm_data_generator.tir.all(x < y, x > z)) == "%s < %s and %s > %s" % (
        x.name,
        y.name,
        x.name,
        z.name,
    )
    assert str(
        gsm_data_generator.tir.all(x < y, y > z + 1, x < z * 2)
    ) == "%s < %s and %s > %s + 1 and %s < %s * 2" % (
        x.name,
        y.name,
        y.name,
        z.name,
        x.name,
        z.name,
    )


def test_bitwise():
    x = te.var("x")
    y = te.var("y")
    assert str(x << y) == "T.shift_left(x, y)"
    assert str(x >> y) == "T.shift_right(x, y)"
    assert str(x & y) == "T.bitwise_and(x, y)"
    assert str(x | y) == "T.bitwise_or(x, y)"
    assert str(x ^ y) == "T.bitwise_xor(x, y)"
    assert str(10 & x) == "T.bitwise_and(10, x)"
    assert str(10 | x) == "T.bitwise_or(10, x)"
    assert str(10 ^ x) == "T.bitwise_xor(10, x)"
    assert str(10 >> x) == "T.shift_right(10, x)"
    assert str(10 << x) == "T.shift_left(10, x)"
    assert str(10 % x) == "10 % x"

    assert str(~x) == "T.bitwise_not(x)"
    assert (gsm_data_generator.tir.const(1, "int8x2") >> 1).dtype == "int8x2"
    assert (x >> gsm_data_generator.tir.const(1, "int32x2")).dtype == "int32x2"
    assert (te.var("z", "int8x2") << gsm_data_generator.tir.const(1, "int8x2")).dtype == "int8x2"


def test_float_bitwise():
    t = gsm_data_generator.tir.const(1.5, dtype="float32")
    for test in [
        lambda lhs, rhs: lhs << rhs,
        lambda lhs, rhs: lhs >> rhs,
        lambda lhs, rhs: lhs | rhs,
        lambda lhs, rhs: lhs ^ rhs,
        lambda lhs, rhs: lhs & rhs,
    ]:
        try:
            test(t, 10.0)
            assert False
        except gsm_data_generator.TVMError:
            pass
    try:
        ~t
        assert False
    except RuntimeError:
        pass


def test_shift_bounds():
    x = te.var("x")
    for test in [lambda lhs, rhs: lhs << rhs, lambda lhs, rhs: lhs >> rhs]:
        # negative case
        for testcase in [(x, -1), (x, 32)]:
            try:
                test(*testcase)
                assert False
            except gsm_data_generator.TVMError:
                pass

        # positive case
        for testcase in [(x, 0), (x, 16), (x, 31)]:
            test(*testcase)


def test_divide_by_zero():
    for test in [
        lambda lhs, rhs: gsm_data_generator.tir.floormod(lhs, rhs),
        lambda lhs, rhs: gsm_data_generator.tir.floordiv(lhs, rhs),
        lambda lhs, rhs: gsm_data_generator.tir.truncmod(lhs, rhs),
        lambda lhs, rhs: gsm_data_generator.tir.truncdiv(lhs, rhs),
        lambda lhs, rhs: gsm_data_generator.tir.div(lhs, rhs),
    ]:
        try:
            test(gsm_data_generator.tir.const(5, "int32"), gsm_data_generator.tir.const(0, "int32"))
            assert False
        except gsm_data_generator.TVMError:
            pass


def test_infinity():
    assert str(gsm_data_generator.tir.infinity("float16")) == 'T.float16("inf")'
    assert str(gsm_data_generator.tir.infinity("float32")) == 'T.float32("inf")'
    assert str(gsm_data_generator.tir.infinity("float64")) == 'T.float64("inf")'


def test_isnan():
    x = te.var("x", "float32")
    assert str(gsm_data_generator.tir.isnan(x)) == "T.isnan(x)"
    assert str(gsm_data_generator.tir.isnan(x).dtype) == "bool"
    y = te.var("y", "float16")
    assert str(gsm_data_generator.tir.isnan(y)) == 'T.isnan(T.Cast("float32", y))'
    z = te.var("z", "int32")
    assert str(gsm_data_generator.tir.isnan(z)) == "T.bool(False)"
    k = te.var("k", "int8x2")
    assert str(gsm_data_generator.tir.isnan(k).dtype) == "uint1x2"


def test_equality():
    a = te.var("a")
    b = te.var("b")
    c = a == b
    assert not c
    d = c != c
    assert not d


def test_equality_string_imm():
    x = "a"
    y = gsm_data_generator.tir.StringImm(x)
    x == y.value
    x == y


def test_prim_func():
    x = te.var("x")
    y = te.var("y")
    b = gsm_data_generator.tir.decl_buffer((x,), "float32")
    stmt = gsm_data_generator.tir.LetStmt(x, 10, gsm_data_generator.tir.Evaluate(x + 1))

    func = gsm_data_generator.tir.PrimFunc([x, y, b], stmt)
    # make sure we can print
    assert func.buffer_map[func.params[2]].same_as(b)

    assert len(func.buffer_map) == 1
    f2 = func.with_attr({"calling_conv": 1, "tir.noalias": True})
    assert f2.attrs["calling_conv"] == 1
    assert not func.attrs


def test_vars():
    x = gsm_data_generator.tir.Var("xyz", "int8")
    assert x.dtype == "int8"
    ptype = gsm_data_generator.ir.PointerType(gsm_data_generator.ir.PrimType("float"))
    x = gsm_data_generator.tir.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert isinstance(ptype.element_type, gsm_data_generator.ir.PrimType)


def test_scoped_storage_vars():
    dtype = "float"
    storage_scope = "global.texture"
    ptype = gsm_data_generator.ir.PointerType(gsm_data_generator.ir.PrimType(dtype), storage_scope)
    x = gsm_data_generator.tir.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert x.type_annotation.storage_scope == storage_scope
    assert isinstance(ptype.element_type, gsm_data_generator.ir.PrimType)


def test_buffer_load_store():
    b = gsm_data_generator.tir.decl_buffer((10,), "float32")
    x = gsm_data_generator.tir.BufferLoad(b, [0])
    assert isinstance(x, gsm_data_generator.tir.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == b
    s = gsm_data_generator.tir.BufferStore(b, 0.1, [0])
    assert isinstance(s, gsm_data_generator.tir.BufferStore)

    s = gsm_data_generator.tir.BufferRealize(b, [gsm_data_generator.ir.Range(0, 1)], True, gsm_data_generator.tir.Evaluate(0))
    assert isinstance(s, gsm_data_generator.tir.BufferRealize)


def test_intimm_cond():
    x = gsm_data_generator.runtime.convert(1)
    y = gsm_data_generator.runtime.convert(1)
    s = {x}
    assert y in s
    assert x == y
    assert x < 20
    assert not (x >= 20)
    assert x < 10 and y < 10
    assert not gsm_data_generator.runtime.convert(x != 1)
    assert x == 1


def _create_ramp(lanes):
    return gsm_data_generator.tir.Ramp(0, 1, lanes)


def _create_broadcast(lanes):
    return gsm_data_generator.tir.Broadcast(0, lanes)


@pytest.mark.parametrize("lanes", [(gsm_data_generator.tir.IntImm(dtype="int64", value=11))])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_lane_types(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.dtype == "int32"
        assert node.lanes == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize("lanes", [(11 * gsm_data_generator.tir.vscale()), (gsm_data_generator.tir.vscale() * 11)])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.a.equal(gsm_data_generator.tir.vscale())
        assert node.lanes.b == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize(
    "lanes", [(gsm_data_generator.tir.vscale()), (gsm_data_generator.tir.vscale() + 3), (gsm_data_generator.tir.vscale() * 2 + 5)]
)
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec_error(lanes, node_func):
    with pytest.raises(gsm_data_generator.error.TVMError):
        node_func(lanes)


def test_broadcast_to_scalable_vec():
    vec = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale()) + 3
    broadcast = vec.b

    assert isinstance(broadcast, gsm_data_generator.tir.expr.Broadcast)
    assert broadcast.value == 3
    assert broadcast.lanes.a.equal(gsm_data_generator.tir.vscale())
    assert broadcast.lanes.b == 4


def test_buffer_load_scalable_vec():
    buf = gsm_data_generator.tir.decl_buffer((24,), "float32")
    index = gsm_data_generator.tir.expr.Ramp(1, 1, 8 * gsm_data_generator.tir.vscale())
    load = gsm_data_generator.tir.BufferLoad(buf, [index])

    assert isinstance(load, gsm_data_generator.tir.BufferLoad)
    assert load.dtype == "float32xvscalex8"


def test_buffer_store_scalable_vec():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    value = gsm_data_generator.tir.expr.Broadcast(1, 4 * gsm_data_generator.tir.vscale())
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    store = gsm_data_generator.tir.BufferStore(b, value, [index])

    assert isinstance(store, gsm_data_generator.tir.BufferStore)
    assert store.value.dtype == "int32xvscalex4"


def test_buffer_store_predicate_invalid_scalability():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    value = gsm_data_generator.tir.expr.Broadcast(1, 4 * gsm_data_generator.tir.vscale())
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    predicate = gsm_data_generator.tir.expr.Broadcast(gsm_data_generator.tir.IntImm("int1", 1), 4)

    err_msg = "Predicate mask dtype and value dtype must both be scalable."
    with pytest.raises(gsm_data_generator.TVMError, match=err_msg):
        gsm_data_generator.tir.BufferStore(b, value, [index], predicate)


def test_buffer_store_predicate_invalid_lanes():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    value = gsm_data_generator.tir.expr.Broadcast(1, 4 * gsm_data_generator.tir.vscale())
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    predicate = gsm_data_generator.tir.expr.Broadcast(gsm_data_generator.tir.IntImm("int1", 1), 8 * gsm_data_generator.tir.vscale())

    err_msg = (
        "Got a predicate mask with 8 lanes, but trying to store a "
        "value with 4 lanes. The number of lanes must match."
    )
    with pytest.raises(gsm_data_generator.TVMError, match=err_msg):
        gsm_data_generator.tir.BufferStore(b, value, [index], predicate)


def test_buffer_store_predicate_elements_invalid_type():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    value = gsm_data_generator.tir.expr.Broadcast(1, 4 * gsm_data_generator.tir.vscale())
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    predicate = gsm_data_generator.tir.expr.Broadcast(1, 4 * gsm_data_generator.tir.vscale())

    err_msg = "Predicate mask elements must be boolean values, but got int32."
    with pytest.raises(gsm_data_generator.TVMError, match=err_msg):
        gsm_data_generator.tir.BufferStore(b, value, [index], predicate)


def test_buffer_load_predicate_elements_invalid_type():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    predicate = gsm_data_generator.tir.expr.Broadcast(1, 4 * gsm_data_generator.tir.vscale())

    err_msg = "Predicate mask elements must be boolean values, but got int32."
    with pytest.raises(gsm_data_generator.TVMError, match=err_msg):
        gsm_data_generator.tir.BufferLoad(b, [index], predicate)


def test_buffer_store_predicate_invalid_scalability():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    predicate = gsm_data_generator.tir.expr.Broadcast(gsm_data_generator.tir.IntImm("int1", 1), 4)

    err_msg = "Predicate mask dtype and load indices must both be scalable."
    with pytest.raises(gsm_data_generator.TVMError, match=err_msg):
        gsm_data_generator.tir.BufferLoad(b, [index], predicate)


def test_buffer_store_predicate_invalid_lanes():
    b = gsm_data_generator.tir.decl_buffer((24,), "int32")
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 4 * gsm_data_generator.tir.vscale())
    predicate = gsm_data_generator.tir.expr.Broadcast(gsm_data_generator.tir.IntImm("int1", 1), 8 * gsm_data_generator.tir.vscale())

    err_msg = (
        "Got a predicate mask with 8 lanes, but trying to load a "
        "vector with 4 lanes. The number of lanes must match."
    )
    with pytest.raises(gsm_data_generator.TVMError, match=err_msg):
        gsm_data_generator.tir.BufferLoad(b, [index], predicate)


def test_scalable_vec_cast():
    b = gsm_data_generator.tir.decl_buffer((24,), "float32")
    value = gsm_data_generator.tir.expr.Broadcast(1, 12 * gsm_data_generator.tir.vscale()).astype("float32xvscalex12")
    index = gsm_data_generator.tir.expr.Ramp(0, 1, 12 * gsm_data_generator.tir.vscale())

    store = gsm_data_generator.tir.BufferStore(b, value, [index])

    assert isinstance(store.value.value, gsm_data_generator.tir.expr.FloatImm)


if __name__ == "__main__":
    gsm_data_generator.testing.main()
