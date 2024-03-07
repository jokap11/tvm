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

import tvm
from tvm import te
import numpy as np
import tvm.relay as relay
from tvm.relay import transform
from tvm.relay.testing.temp_op_attr import TempOpAttr



def test_reduced_input_nchw_run():
    channels =  tvm.tir.IntImm(value=3, dtype="int8")
    data_shape = (1, 3, 4, 4)
    weight_shape = (1,3,2,2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2),1, channels, weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.ones(shape=data_shape, dtype=np.int8)
    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = [[[[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]]]]

    tvm.testing.assert_allclose(out, ref)


def test_reduced_input_nchw_run_single():
    channels =  tvm.tir.IntImm(value=3, dtype="int8")
    data_shape = (1, 1, 4, 4)
    weight_shape = (1,1,2,2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2),1, channels, weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.ones(shape=data_shape, dtype=np.int8)
    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = [[ [[4, 4], [4, 4]]]]

    tvm.testing.assert_allclose(out, ref)

def test_reduced_input_nchw_asymmetric_run():
    channels =  tvm.tir.IntImm(value=2, dtype="int8")
    data_shape = (1, 2, 4, 6)
    weight_shape = (1, 2, 2, 2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2),1, channels, weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.array([[[[1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]],
                        [[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = [[[[7, 6], [6, 6]],
            [[6, 6], [6, 6]]]]

    tvm.testing.assert_allclose(out, ref)

def test_reduced_input_nchw_asymmetric_run_2():
    channels =  tvm.tir.IntImm(value=3, dtype="int8")
    data_shape = (1, 3, 9, 6)
    weight_shape = (1, 3, 3, 3)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(3,3),1, channels, weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.array([[[[1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]],
                        [[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]],
                        [[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = [[[[6, 7, 6], [6, 6, 6], [6, 6, 6]],
            [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
            [[6, 6, 6], [6, 6, 6], [6, 6, 6]],]]

    tvm.testing.assert_allclose(out, ref)

if __name__ == "__main__":
    test_reduced_input_nchw_run()
    test_reduced_input_nchw_run_single()
    test_reduced_input_nchw_asymmetric_run()
    test_reduced_input_nchw_asymmetric_run_2()