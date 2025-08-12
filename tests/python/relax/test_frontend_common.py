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
from gsmDataGen.relax.frontend import detach_params
from gsmDataGen.script.parser import relax as R


def test_detach_params():
    @R.function
    def func(x: R.Tensor((2, 3), "float32")):
        return x

    param = gsmDataGen.nd.empty((3,), "float32")
    mod = gsmDataGen.IRModule({"func": func.with_attr("params", [param])})
    detached_mod, detached_params = detach_params(mod)

    gsmDataGen.ir.assert_structural_equal(detached_mod, gsmDataGen.IRModule({"func": func}))
    assert len(detached_params) == 1
    assert "func" in detached_params
    assert isinstance(detached_params["func"], list)
    assert len(detached_params["func"]) == 1
    gsmDataGen.testing.assert_allclose(detached_params["func"][0].numpy(), param.numpy())


if __name__ == "__main__":
    gsmDataGen.testing.main()
