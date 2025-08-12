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
from gsm_data_generator import TVMError
from gsm_data_generator.script import tir as T


class BaseBeforeAfter(gsm_data_generator.testing.CompareBeforeAfter):
    @gsm_data_generator.testing.fixture
    def transform(self):
        return gsm_data_generator.tir.transform.RemoveAssume()


class TestRemoveAssume(BaseBeforeAfter):
    """Remove any instance of T.assume"""

    def before(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 5))
        A[0] = 10

    def expected(A: T.Buffer(1, "int32")):
        A[0] = 10


class TestRemoveAssumeLoop(BaseBeforeAfter):
    """Loops containing only T.assume should be removed"""

    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            A[i] = 10

    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 10


if __name__ == "__main__":
    gsm_data_generator.testing.main()
