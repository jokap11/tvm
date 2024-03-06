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
# pylint: disable=invalid-name
"""TVM operator reduced_input compute."""
from __future__ import absolute_import
from . import cpp


def reduced_input(data, strides: list, group: int, channels, 
                  weight_shape: list,  kernel_layout:str, data_layout:str):
    """Applies the reduced input checksum opertaion on the input array (Hari et al).

    Reduces data dimension to the according filter dimension for non unit strided conv2D
    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.
    strides: list, 
        strides of original conv2d -> extended to 4D due to strided slicing 
    group: int, 
        split data into groups from original convolutions
    channels: int, 
        output convolutions (only used for simplified depth-wise conv2d check)
    weight_shape: list, 
        weight shape required to infer output dimension (use filter-wise sum size ;))
    kernel_layout:str, 
        Kernel layout required to check on weight shape
    data_layout:str
        Data layout of original conv2d required to interpret data shape
    Returns
    -------
    result : tvm.te.Tensor
        The result of dropout
    """
    return cpp.nn.reduced_input(data, strides, group, channels, weight_shape,  kernel_layout, data_layout)
