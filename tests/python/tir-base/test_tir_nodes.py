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
import gsmDataGen
from gsmDataGen import ir, te


def test_const():
    x = gsmDataGen.tir.const(1, "int32")
    assert x.dtype == "int32"
    assert isinstance(x, gsmDataGen.tir.IntImm)


def test_te_const():
    x = gsmDataGen.te.const(1, "int32")
    assert x.dtype == "int32"
    assert isinstance(x, gsmDataGen.tir.IntImm)


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
        assert gsmDataGen.tir.const(data).dtype == str(np.array(data).dtype)

    assert gsmDataGen.tir.const(True).dtype == "bool"
    assert gsmDataGen.tir.const(1).dtype == "int32"
    assert gsmDataGen.tir.const(1.0).dtype == "float32"


def test_make():
    x = gsmDataGen.tir.const(1, "int32")
    y = te.var("x")
    z = x + y
    assert isinstance(gsmDataGen.tir.max(x, y), gsmDataGen.tir.Max)
    assert isinstance(gsmDataGen.tir.min(x, y), gsmDataGen.tir.Min)


def test_ir():
    x = gsmDataGen.tir.const(1, "int32")
    y = gsmDataGen.tir.IntImm("int32", 1)
    z = x + y
    stmt = gsmDataGen.tir.Evaluate(z)
    assert isinstance(stmt, gsmDataGen.tir.Evaluate)


def test_ir2():
    buf_size = te.var("size")
    x = te.var("n")

    storage_type = ir.PrimType("int32")
    handle_type = ir.PointerType(storage_type)
    array = te.var("array", handle_type)
    buf = gsmDataGen.tir.decl_buffer([buf_size], "int32", data=array)

    st = gsmDataGen.tir.BufferStore(buf, x + 1, [1])
    assert isinstance(st, gsmDataGen.tir.BufferStore)
    assert st.buffer == buf
    assert st.buffer.data == array


def test_let():
    x = te.var("x")
    y = te.var("y")
    stmt = gsmDataGen.tir.LetStmt(x, 10, gsmDataGen.tir.Evaluate(x + 1))


def test_cast():
    x = te.var("x", dtype="float32")
    y = x.astype("int32")
    z = x.astype("float32x4")
    assert isinstance(y, gsmDataGen.tir.Cast)
    assert isinstance(z, gsmDataGen.tir.Broadcast)
    assert z.lanes == 4

    s = gsmDataGen.tir.StringImm("s")
    with pytest.raises(gsmDataGen.error.TVMError):
        try:
            s.astype("int")
        except Exception as e:
            assert "Can't cast a handle to other types" in str(e)
            raise


def test_attr():
    x = te.var("x")
    y = te.var("y")
    stmt = gsmDataGen.tir.AttrStmt(y, "stride", 10, gsmDataGen.tir.Evaluate(x + 1))
    assert stmt.node == y

    a = gsmDataGen.runtime.convert(1)
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
    x = gsmDataGen.tir.Evaluate(0)
    gsmDataGen.tir.For(te.var("i"), 0, 1, gsmDataGen.tir.ForKind.SERIAL, x)


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
        gsmDataGen.tir.any()
        assert False
    except ValueError:
        pass
    assert str(gsmDataGen.tir.any(x < y)) == "%s < %s" % (x.name, y.name)
    assert str(gsmDataGen.tir.any(x < y, x > z)) == "%s < %s or %s > %s" % (
        x.name,
        y.name,
        x.name,
        z.name,
    )
    assert str(
        gsmDataGen.tir.any(x < y, y > z + 1, x < z * 2)
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
        gsmDataGen.tir.all()
        assert False
    except ValueError:
        pass
    assert str(gsmDataGen.tir.all(x < y)) == "%s < %s" % (x.name, y.name)
    assert str(gsmDataGen.tir.all(x < y, x > z)) == "%s < %s and %s > %s" % (
        x.name,
        y.name,
        x.name,
        z.name,
    )
    assert str(
        gsmDataGen.tir.all(x < y, y > z + 1, x < z * 2)
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
    assert (gsmDataGen.tir.const(1, "int8x2") >> 1).dtype == "int8x2"
    assert (x >> gsmDataGen.tir.const(1, "int32x2")).dtype == "int32x2"
    assert (te.var("z", "int8x2") << gsmDataGen.tir.const(1, "int8x2")).dtype == "int8x2"


def test_float_bitwise():
    t = gsmDataGen.tir.const(1.5, dtype="float32")
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
        except gsmDataGen.TVMError:
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
            except gsmDataGen.TVMError:
                pass

        # positive case
        for testcase in [(x, 0), (x, 16), (x, 31)]:
            test(*testcase)


def test_divide_by_zero():
    for test in [
        lambda lhs, rhs: gsmDataGen.tir.floormod(lhs, rhs),
        lambda lhs, rhs: gsmDataGen.tir.floordiv(lhs, rhs),
        lambda lhs, rhs: gsmDataGen.tir.truncmod(lhs, rhs),
        lambda lhs, rhs: gsmDataGen.tir.truncdiv(lhs, rhs),
        lambda lhs, rhs: gsmDataGen.tir.div(lhs, rhs),
    ]:
        try:
            test(gsmDataGen.tir.const(5, "int32"), gsmDataGen.tir.const(0, "int32"))
            assert False
        except gsmDataGen.TVMError:
            pass


def test_infinity():
    assert str(gsmDataGen.tir.infinity("float16")) == 'T.float16("inf")'
    assert str(gsmDataGen.tir.infinity("float32")) == 'T.float32("inf")'
    assert str(gsmDataGen.tir.infinity("float64")) == 'T.float64("inf")'


def test_isnan():
    x = te.var("x", "float32")
    assert str(gsmDataGen.tir.isnan(x)) == "T.isnan(x)"
    assert str(gsmDataGen.tir.isnan(x).dtype) == "bool"
    y = te.var("y", "float16")
    assert str(gsmDataGen.tir.isnan(y)) == 'T.isnan(T.Cast("float32", y))'
    z = te.var("z", "int32")
    assert str(gsmDataGen.tir.isnan(z)) == "T.bool(False)"
    k = te.var("k", "int8x2")
    assert str(gsmDataGen.tir.isnan(k).dtype) == "uint1x2"


def test_equality():
    a = te.var("a")
    b = te.var("b")
    c = a == b
    assert not c
    d = c != c
    assert not d


def test_equality_string_imm():
    x = "a"
    y = gsmDataGen.tir.StringImm(x)
    x == y.value
    x == y


def test_prim_func():
    x = te.var("x")
    y = te.var("y")
    b = gsmDataGen.tir.decl_buffer((x,), "float32")
    stmt = gsmDataGen.tir.LetStmt(x, 10, gsmDataGen.tir.Evaluate(x + 1))

    func = gsmDataGen.tir.PrimFunc([x, y, b], stmt)
    # make sure we can print
    assert func.buffer_map[func.params[2]].same_as(b)

    assert len(func.buffer_map) == 1
    f2 = func.with_attr({"calling_conv": 1, "tir.noalias": True})
    assert f2.attrs["calling_conv"] == 1
    assert not func.attrs


def test_vars():
    x = gsmDataGen.tir.Var("xyz", "int8")
    assert x.dtype == "int8"
    ptype = gsmDataGen.ir.PointerType(gsmDataGen.ir.PrimType("float"))
    x = gsmDataGen.tir.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert isinstance(ptype.element_type, gsmDataGen.ir.PrimType)


def test_scoped_storage_vars():
    dtype = "float"
    storage_scope = "global.texture"
    ptype = gsmDataGen.ir.PointerType(gsmDataGen.ir.PrimType(dtype), storage_scope)
    x = gsmDataGen.tir.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert x.type_annotation.storage_scope == storage_scope
    assert isinstance(ptype.element_type, gsmDataGen.ir.PrimType)


def test_buffer_load_store():
    b = gsmDataGen.tir.decl_buffer((10,), "float32")
    x = gsmDataGen.tir.BufferLoad(b, [0])
    assert isinstance(x, gsmDataGen.tir.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == b
    s = gsmDataGen.tir.BufferStore(b, 0.1, [0])
    assert isinstance(s, gsmDataGen.tir.BufferStore)

    s = gsmDataGen.tir.BufferRealize(b, [gsmDataGen.ir.Range(0, 1)], True, gsmDataGen.tir.Evaluate(0))
    assert isinstance(s, gsmDataGen.tir.BufferRealize)


def test_intimm_cond():
    x = gsmDataGen.runtime.convert(1)
    y = gsmDataGen.runtime.convert(1)
    s = {x}
    assert y in s
    assert x == y
    assert x < 20
    assert not (x >= 20)
    assert x < 10 and y < 10
    assert not gsmDataGen.runtime.convert(x != 1)
    assert x == 1


def _create_ramp(lanes):
    return gsmDataGen.tir.Ramp(0, 1, lanes)


def _create_broadcast(lanes):
    return gsmDataGen.tir.Broadcast(0, lanes)


@pytest.mark.parametrize("lanes", [(gsmDataGen.tir.IntImm(dtype="int64", value=11))])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_lane_types(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.dtype == "int32"
        assert node.lanes == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize("lanes", [(11 * gsmDataGen.tir.vscale()), (gsmDataGen.tir.vscale() * 11)])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.a.equal(gsmDataGen.tir.vscale())
        assert node.lanes.b == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize(
    "lanes", [(gsmDataGen.tir.vscale()), (gsmDataGen.tir.vscale() + 3), (gsmDataGen.tir.vscale() * 2 + 5)]
)
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec_error(lanes, node_func):
    with pytest.raises(gsmDataGen.error.TVMError):
        node_func(lanes)


def test_broadcast_to_scalable_vec():
    vec = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale()) + 3
    broadcast = vec.b

    assert isinstance(broadcast, gsmDataGen.tir.expr.Broadcast)
    assert broadcast.value == 3
    assert broadcast.lanes.a.equal(gsmDataGen.tir.vscale())
    assert broadcast.lanes.b == 4


def test_buffer_load_scalable_vec():
    buf = gsmDataGen.tir.decl_buffer((24,), "float32")
    index = gsmDataGen.tir.expr.Ramp(1, 1, 8 * gsmDataGen.tir.vscale())
    load = gsmDataGen.tir.BufferLoad(buf, [index])

    assert isinstance(load, gsmDataGen.tir.BufferLoad)
    assert load.dtype == "float32xvscalex8"


def test_buffer_store_scalable_vec():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    value = gsmDataGen.tir.expr.Broadcast(1, 4 * gsmDataGen.tir.vscale())
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    store = gsmDataGen.tir.BufferStore(b, value, [index])

    assert isinstance(store, gsmDataGen.tir.BufferStore)
    assert store.value.dtype == "int32xvscalex4"


def test_buffer_store_predicate_invalid_scalability():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    value = gsmDataGen.tir.expr.Broadcast(1, 4 * gsmDataGen.tir.vscale())
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    predicate = gsmDataGen.tir.expr.Broadcast(gsmDataGen.tir.IntImm("int1", 1), 4)

    err_msg = "Predicate mask dtype and value dtype must both be scalable."
    with pytest.raises(gsmDataGen.TVMError, match=err_msg):
        gsmDataGen.tir.BufferStore(b, value, [index], predicate)


def test_buffer_store_predicate_invalid_lanes():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    value = gsmDataGen.tir.expr.Broadcast(1, 4 * gsmDataGen.tir.vscale())
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    predicate = gsmDataGen.tir.expr.Broadcast(gsmDataGen.tir.IntImm("int1", 1), 8 * gsmDataGen.tir.vscale())

    err_msg = (
        "Got a predicate mask with 8 lanes, but trying to store a "
        "value with 4 lanes. The number of lanes must match."
    )
    with pytest.raises(gsmDataGen.TVMError, match=err_msg):
        gsmDataGen.tir.BufferStore(b, value, [index], predicate)


def test_buffer_store_predicate_elements_invalid_type():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    value = gsmDataGen.tir.expr.Broadcast(1, 4 * gsmDataGen.tir.vscale())
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    predicate = gsmDataGen.tir.expr.Broadcast(1, 4 * gsmDataGen.tir.vscale())

    err_msg = "Predicate mask elements must be boolean values, but got int32."
    with pytest.raises(gsmDataGen.TVMError, match=err_msg):
        gsmDataGen.tir.BufferStore(b, value, [index], predicate)


def test_buffer_load_predicate_elements_invalid_type():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    predicate = gsmDataGen.tir.expr.Broadcast(1, 4 * gsmDataGen.tir.vscale())

    err_msg = "Predicate mask elements must be boolean values, but got int32."
    with pytest.raises(gsmDataGen.TVMError, match=err_msg):
        gsmDataGen.tir.BufferLoad(b, [index], predicate)


def test_buffer_store_predicate_invalid_scalability():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    predicate = gsmDataGen.tir.expr.Broadcast(gsmDataGen.tir.IntImm("int1", 1), 4)

    err_msg = "Predicate mask dtype and load indices must both be scalable."
    with pytest.raises(gsmDataGen.TVMError, match=err_msg):
        gsmDataGen.tir.BufferLoad(b, [index], predicate)


def test_buffer_store_predicate_invalid_lanes():
    b = gsmDataGen.tir.decl_buffer((24,), "int32")
    index = gsmDataGen.tir.expr.Ramp(0, 1, 4 * gsmDataGen.tir.vscale())
    predicate = gsmDataGen.tir.expr.Broadcast(gsmDataGen.tir.IntImm("int1", 1), 8 * gsmDataGen.tir.vscale())

    err_msg = (
        "Got a predicate mask with 8 lanes, but trying to load a "
        "vector with 4 lanes. The number of lanes must match."
    )
    with pytest.raises(gsmDataGen.TVMError, match=err_msg):
        gsmDataGen.tir.BufferLoad(b, [index], predicate)


def test_scalable_vec_cast():
    b = gsmDataGen.tir.decl_buffer((24,), "float32")
    value = gsmDataGen.tir.expr.Broadcast(1, 12 * gsmDataGen.tir.vscale()).astype("float32xvscalex12")
    index = gsmDataGen.tir.expr.Ramp(0, 1, 12 * gsmDataGen.tir.vscale())

    store = gsmDataGen.tir.BufferStore(b, value, [index])

    assert isinstance(store.value.value, gsmDataGen.tir.expr.FloatImm)


if __name__ == "__main__":
    gsmDataGen.testing.main()
