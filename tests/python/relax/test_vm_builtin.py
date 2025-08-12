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
import gsm_data_generator.script
import gsm_data_generator.testing
from gsm_data_generator import relax
from gsm_data_generator.script import relax as R, ir as I


def test_multinomial_from_uniform():
    @I.ir_module
    class CallSample:
        @R.function
        def foo(x: R.Tensor((3, 5), "float32"), y: R.Tensor((3, 1), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.multinomial_from_uniform",
                x,
                y,
                sinfo_args=(R.Tensor((3, 1), dtype="int64")),
            )
            return z

    mod = CallSample
    target = gsm_data_generator.target.Target("llvm", host="llvm")
    ex = gsm_data_generator.compile(mod, target)
    np_rand = np.random.rand(3, 5).astype(np.float32)
    # normalize it to get the random prob
    np_prob = np_rand / np_rand.sum(axis=1, keepdims=True)
    nd_prob = gsm_data_generator.nd.array(np_prob)
    # special sample to get deterministic results
    nd_sample = gsm_data_generator.nd.array(np.array([[1.0], [0], [1]]).astype(np.float32))

    vm = relax.VirtualMachine(ex, gsm_data_generator.cpu())
    res = vm["foo"](nd_prob, nd_sample)
    gsm_data_generator.testing.assert_allclose(res.numpy(), np.array([[4], [0], [4]]).astype(np.int64))


@gsm_data_generator.testing.parametrize_targets("cuda")
def test_alloc_tensor_raises_out_of_memory(target, dev):
    """Out-of-memory exceptions may be raised from VM

    This is a regression test.  In previous implementations, the Relax
    VM would segfault if the built-in function
    "vm.builtin.alloc_storage" was unable to allocate the requested
    buffer.
    """

    @I.ir_module
    class Module:
        @R.function
        def main():
            # Allocate a 1-petabyte tensor to trigger OOM.  If the CI
            # ever runs on a device with more than 1 petabyte of GPU
            # memory, this test will need to be updated.
            output = R.builtin.alloc_tensor(R.shape([1024, 1024, 1024, 1024, 1024]), "uint8", 0)
            return output

    built = gsm_data_generator.compile(Module, target=target)
    vm = relax.VirtualMachine(built, dev)

    with pytest.raises(Exception, match="CUDA: out of memory"):
        vm["main"]()


if __name__ == "__main__":
    gsm_data_generator.testing.main()
