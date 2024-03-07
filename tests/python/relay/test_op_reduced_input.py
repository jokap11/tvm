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
import numpy as np
import tvm.relay as relay
import tvm.testing
from tvm import relay
import tvm.contrib



def test_reduced_input_nchw_run():
    data_shape = (1, 3, 4, 4)
    weight_shape = (1,3,2,2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2), weight_shape, "OIHW", "NCHW")
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

    np.testing.assert_equal(out, np.int32(ref))


def test_reduced_input_nchw_run_single():
    data_shape = (1, 1, 4, 4)
    weight_shape = (1,1,2,2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2), weight_shape, "OIHW", "NCHW")
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

    np.testing.assert_equal(out, np.int32(ref))


def test_reduced_input_nchw_asymmetric_run_1():
    data_shape = (1, 2, 4, 6)
    weight_shape = (1, 2, 2, 2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2), weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.array([[[[1, 2, 3, 1, 2, 1],
                         [1, 1, 1, 1, 2, 1],
                         [1, 3, 0, 6, 2, 1],
                         [1, -1, 0, 1, 1, 0]],
                        [[1,   33,  1,  42, -5,  69],
                         [1,   4,   13, 2,  13,  18],
                         [14, -111, 1,  1,  22,  1],
                         [1,   31,  12, 14, 23,   1]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    print(out)
    print(type(out))
    ref = [[[[9, 14],   [6, 3]],
            [[34, 35], [63, 70]]]]
    np.testing.assert_equal(out, np.int32(ref))

def test_reduced_input_nchw_asymmetric_stride_run():
    data_shape = (1, 2, 4, 6)
    weight_shape = (1, 2, 2, 3)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,3), weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.array([[[[1,  2,   3,   1, 2, 12],
                         [1,  13, 11,  13, 2, 1],
                         [1,  3,  20,   6, 2, 1],
                         [1, -1,   0, -14, 1, 0]],
                        [[1,   33,  1,  42, -5,  69],
                         [1,   4,   13, 2,  13,  18],
                         [14, -111, 1,  1,  22,  1],
                         [1,   31,  12, 14, 23,   1]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    print(out)
    print(type(out))
    ref = [[[[9,  9, 36],  [1, 15, 12]],
            [[58, -61, 72],[18, 71,44]]]]
    np.testing.assert_equal(out, np.int32(ref))



def test_reduced_input_nchw_overflow():
    data_shape = (1, 1, 4, 6)
    weight_shape = (1, 1, 2, 2)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2), weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.array([[[[100, 2,  44,  1,    33,  1],
                         [1,   13, 1,   13,   22,  111],
                         [-21, 30, 100, 64,   20,  31],
                         [1,  -13, 100, 122,  12,  120]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    print(out)
    print(type(out))
    ref = [[[[276, 129],[137, 366]]]]
    np.testing.assert_equal(out, np.int32(ref))

def test_reduced_input_nchw_asymmetric_run_2():
    data_shape = (1, 3, 9, 6)
    weight_shape = (1, 3, 3, 3)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(3,3), weight_shape, "OIHW", "NCHW")
    func = relay.Function([x], inp)
    x_data = np.array([[[[  44, -118,   -1,   12,  -81,   42],
                        [  68,   23,  -11,   38, -106,   55],
                        [  64,   76,  -95,   88,  -61,   51],
                        [ -50,   26,  123,  -46,   34,   91],
                        [  67,  -10,   -3,   11,  -25,   -3],
                        [ 101,   88, -119,   36,  -12,  -20],
                        [  83,   94,   33,   31, -107,  -47],
                        [ -39,   37,  114,   86,  -26,  -30],
                        [ -92,   55, -123,  -16,  -41,  -70]],
                        [[ -85,  -52,  -58,  -68,  -53,  100],
                        [  88,   61,    4, -114,  -40,   26],
                        [  50,  118,   12,   77,   76,  -59],
                        [ -70,  -71,  -87,  -30,   65,  -62],
                        [ -56,   -6,  102,   -3,   46,   74],
                        [ -89,  -54,  106,   79,  -41,   40],
                        [ -27,    7,   46,   72,   95,   -6],
                        [ -40,  -34,  -21,   17,  -47,   11],
                        [  13,  -28,   37,  102,  115,  108]],
                        [[-103,  -62, -119,   86,  -51,  -21],
                        [ -81, -110,  -56,   24,  -33,  -42],
                        [-119, -101,  -50, -106,   20,   23],
                        [ 110,   37,  -13, -120,  -83, -123],
                        [  80, -109,  -59,  -34,  115,   18],
                        [ -90,   85,   69,  -98, -120,  117],
                        [ 126,  -57,  -75,   71,  -49, -121],
                        [ 110,  -93,   47,  -99,   57,   94],
                        [  64, -119,   62,   94,  -46,   59]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = np.array([[[[  74, -152,  241],
            [ 231, -107,  122],
            [ 181,  105, -376]],
            [[-208,   -9,  -67],
            [-108,  -20,  196],
            [ 232,  186,  244]],
            [[ 170, -265, -472],
            [   0, -173,    2],
            [-255, -281,  280]]]]).astype("int32")
    np.testing.assert_equal(out, np.int32(ref))


def test_reduced_input_nhwc_run_1():
    data_shape = (1, 6, 3, 2)
    weight_shape = (3, 3, 2, 1)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(3,3), weight_shape, "HWIO", "NHWC")
    func = relay.Function([x], inp)
    x_data = np.array(
        [[[[1, 12],
            [11, 122],
            [21, 66]],
            [[31, 12],
            [41, 33],
            [51, 12]],
            [[-1, -120],
            [14, 12],
            [44, 12]],
            [[2, 12],
            [12, 122],
            [22, -66]],
            [[32, 12],
            [42, 33],
            [52, 12]],
            [[0, -120],
            [4, 12],
            [44, 12]]]]).astype("int8")

    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = np.array([[[[ 3, 24],
                    [23, 244],
                    [43, 0]],
                    [[63, 24],
                    [83, 66],
                    [103, 24]],
                    [[-1, -240],
                    [18, 24],
                    [88, 24]]]]).astype("int32")
    np.testing.assert_equal(out, np.int32(ref))


def test_reduced_input_nhwc_run_2():
    data_shape = (1, 4, 4, 2)
    weight_shape = (2, 2, 2, 1)
    x  = relay.var("data", shape=data_shape, dtype="int8")
    inp = relay.nn.reduced_input(x,(2,2), weight_shape, "HWIO", "NHWC")
    func = relay.Function([x], inp)
    x_data = np.array([[[[1, 12],
                        [2, 122],
                        [3, 13],
                        [4, 66]],
                        [[5, 12],
                        [6, 33],
                        [7, 33],
                        [8, 12]],
                        [[9, -120],
                        [10, 12],
                        [11, 12],
                        [12, 12]],
                        [[13, 12],
                        [14, 122],
                        [15, 122],
                        [16, -66]]]]).astype("int8")
    mod = tvm.IRModule.from_expr(func)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)
    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", x_data)
    runtime.run()
    out = runtime.get_output(0).numpy()
    ref = np.array([[[[ 24, -83],
                    [28, 212]],
                    [[40, 179],
                    [44, 101]]]]).astype("int32")
    np.testing.assert_equal(out, np.int32(ref))




if __name__ == "__main__":
    test_reduced_input_nchw_run()
    test_reduced_input_nchw_run_single()
    test_reduced_input_nchw_asymmetric_stride_run()
    test_reduced_input_nchw_overflow()
    test_reduced_input_nhwc_run_1()
    test_reduced_input_nhwc_run_2()
    test_reduced_input_nchw_asymmetric_run_1()
    test_reduced_input_nchw_asymmetric_run_2()