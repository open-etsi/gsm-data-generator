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
import pytest
import numpy as np
import gsm_data_generator
import gsm_data_generator.testing

from gsm_data_generator import relax
from gsm_data_generator.script import relax as R
from gsm_data_generator.relax.dpl import make_fused_bias_activation_pattern
from gsm_data_generator.contrib.pickle_memoize import memoize


@gsm_data_generator.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2)

        return conv2


has_dnnl = gsm_data_generator.get_global_func("relax.ext.dnnl", True)

dnnl_enabled = pytest.mark.skipif(
    not has_dnnl,
    reason="DNNL note enabled.",
)

pytestmark = [dnnl_enabled]


def build_and_run(mod, inputs, legalize=False):
    target = gsm_data_generator.target.Target("llvm")
    dev = gsm_data_generator.cpu()
    inputs = [gsm_data_generator.nd.array(inp, dev) for inp in inputs]

    with gsm_data_generator.transform.PassContext(config={"relax.transform.apply_legalize_ops": legalize}):
        ex = gsm_data_generator.compile(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    return f(*inputs).numpy()


def test_dnnl_offload():
    pat = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=False, activation="relax.nn.relu"
    )

    seq = gsm_data_generator.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern([("dnnl.conv2d_relu", pat)]),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )

    @memoize("relax.tests.test_codegen_dnnl.conv2d_relu_x2")
    def get_ref():
        data_np = np.random.randn(1, 64, 56, 56).astype("float32")
        weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
        weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")
        inputs = [data_np, weight1_np, weight2_np]
        ref = build_and_run(Conv2dReLUx2, inputs, legalize=True)
        return inputs, ref

    inputs, ref = get_ref()

    out = build_and_run(seq(Conv2dReLUx2), inputs)

    gsm_data_generator.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_dnnl_offload()
