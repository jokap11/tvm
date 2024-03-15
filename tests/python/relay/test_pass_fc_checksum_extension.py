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
# pylint: disable=invalid-name,too-many-locals,too-many-arguments,missing-module-docstring
# Changed by Kappes Johannes @2024

import tvm
from tvm import relay
from tvm.relay import transform
import numpy as np
from tvm.contrib import graph_executor


def run_opt_pass(expr, opt_pass):
    "runs the opt_pass on the expr of a function the function"
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    return mod["main"]




def test_single_dense_checksum():
    """Simple testcase."""

    def before(x, w1):
        args = [x, w1]
        y = relay.nn.dense(x, w1, out_dtype="int32")
        return relay.Function(args, y)

    def expected(x, w1):
        args = [x, w1]
        fc = relay.nn.dense(x, w1, out_dtype="int32")
        fc_32 = relay.cast(fc, "int32")
        col_sum_fc = relay.sum(fc_32, 0, True)
        x_32 = relay.cast(x, "int32")
        col_sum_x = relay.sum(x_32, 0, True)
        lhs_1 = relay.nn.dense(col_sum_x, w1, out_dtype="int32")
        result_1 = relay.not_equal(lhs_1, col_sum_fc)
        result_1_int8 = relay.cast(result_1, "int8")
        result_neq = relay.sum(result_1_int8, axis=(0, 1)) #sum up all boolean values
        row_sum_fc = relay.sum(fc_32, 1, True)
        w_32 = relay.cast(w1, "int32")
        row_sum_w = relay.sum(w_32, 0, True) #transposed and we use W instead of W^T
        lhs_2 = relay.nn.dense(x, row_sum_w, out_dtype="int32")
        result_2 = relay.not_equal(lhs_2, row_sum_fc)
        result_2_int8 = relay.cast(result_2, "int8")
        result_neq2 = relay.sum(result_2_int8, axis=(0, 1)) #sum up all boolean values
        y = relay.Tuple([fc, result_neq, result_neq2])
        return relay.Function(args, y)
        
        

    def check(x_shape, w_shape):
        x = relay.var("x", shape=x_shape, dtype="int8")
        w1 = relay.var("w1", shape=w_shape, dtype="int8")
        y_expected = expected(x, w1)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        print(y_expected)
        y_before = before(x, w1)
        y = run_opt_pass(y_before, transform.ExtendDense())
        y = run_opt_pass(y, transform.InferType())
        print("After pass:")
        print(y)
        print(tvm.ir.base.get_first_structural_mismatch(y, y_expected))
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

#Calculate dimension of ones tensor for Input checksum calc 

    check((1, 4),(4, 4))
    check((2, 4),(4, 4))
    check((5, 7),(3, 7))
    check((3, 12),(2, 12))
    check((17, 12),(2, 12))
    check((13, 27),(5, 27))




def verify(data_shape, data_dtype, kernel_shape, kernel_dtype,out_dtype):
    def get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype):
        # Keeping inputs multiple of 4 because of a bug in Average Pool2d
        # https://discuss.tvm.apache.org/t/pool2d-gives-bad-output-for-integer-inputs/3377

        low_dict  = {"int8": -128,"uint8": 0, "int16": -32768, "uint16":0}
        high_dict = {"int8": 127,"uint8": 255, "int16": 32767, "uint16":65535}
        golden_data = np.random.randint(low=low_dict[data_dtype],
                            high=high_dict[data_dtype], size=data_shape).astype(data_dtype)
        golden_weight = np.random.randint(low=low_dict[kernel_dtype],
                            high=high_dict[kernel_dtype], size=kernel_shape).astype(kernel_dtype)

        return (golden_data, golden_weight)

    def get_output(func, golden_inputs):
        with tvm.transform.PassContext(opt_level=2):
            golden_data, golden_weight = golden_inputs
            params = {"kernel": golden_weight}
            graph, lib, params = relay.build(func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod
            return res

    def create_dense(data_shape, data_dtype, kernel_shape, kernel_dtype, out_dtype):
        x  = relay.var("data", shape=data_shape, dtype=data_dtype)
        w1 = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)
        args = [x,w1]
        y = relay.nn.dense(x, w1, out_dtype=out_dtype)
        return relay.Function(args, y)

    golden_inputs = get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype)
    ref_func = create_dense(data_shape, data_dtype, kernel_shape, kernel_dtype,out_dtype)
    ref_func = run_opt_pass(ref_func, transform.ExtendDense())
    ref_func = run_opt_pass(ref_func, transform.InferType())
    golden_output = get_output(ref_func, golden_inputs)
    # Test if checksum unequivalence is wrong = intended
    np.testing.assert_equal(golden_output.get_output(1).numpy(), np.array(False))
    np.testing.assert_equal(golden_output.get_output(2).numpy(), np.array(False))

def test_output():
    verify((1, 37),"int8",(1, 37),"int8", "int32")
    verify((2, 17),"int8",(1, 17),"int8", "int32")
    verify((3, 44),"int8",(2, 44),"int8", "int32")
    verify((5, 373),"int8",(12, 373),"int8", "int32")
    verify((1, 23,),"int8",(1, 23),"int8", "int32")
    verify((83, 69),"uint8",(1, 69),"uint8", "uint32")
    verify((20, 7),"uint8",(12, 7),"uint8", "uint32")


if __name__ == "__main__":
    test_single_dense_checksum()
    test_output()