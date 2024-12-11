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
from tvm import  te
from ..utils import get_const_tuple



def reduced_input(data: te.Tensor, strides, weight_shape, kernel_layout, data_layout, mode):
    """Applies the reduced input checksum opertaion on the input array (Hari et al).

    Reduces data dimension to the according filter dimension for non unit strided conv2D
    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.
    strides: list,
        strides of original conv2d -> extended to 4D due to strided slicing
    weight_shape: list,
        weight shape required to infer output dimension (use filter-wise checksum sum size ;))
    kernel_layout:str,
        Kernel layout required to check on weight shape
    data_layout:str,
        Data layout of original conv2d required to interpret data shape
    mode: str
        Used to interpret weigh_shape dimensions (depth/group vs standard conv)
    Returns
    -------
    result : tvm.te.Tensor
        The result of reduced_input with data_layout
    """


    weight_height = weight_shape[kernel_layout.find("H")]
    weight_width = weight_shape[kernel_layout.find("W")]
    stride_h, stride_w = strides

    input_dimensions = get_const_tuple(data.shape)

    batch_size = 1

    if mode == "depth": #Groups=#Filter N==O
        channels =  weight_shape[kernel_layout.find("O")]
    else: #Channels=Channeös C==I
        channels =  weight_shape[kernel_layout.find("I")]

    image_height = input_dimensions[data_layout.find("H")]
    image_width = input_dimensions[data_layout.find("W")]
    
    n_ax = te.reduce_axis((0, batch_size), name="n_ax")
    h_ax = te.reduce_axis((0, (image_height - weight_height) // stride_h + 1), name="h_ax")
    w_ax = te.reduce_axis((0, (image_width - weight_width) // stride_w + 1), name="w_ax")




    if data_layout == "NCHW":
        output = te.compute(
            (batch_size, channels, weight_height, weight_width),
            lambda n, c, h, w: te.sum(data[n+n_ax, c, h+stride_h*h_ax , w+stride_w*w_ax].astype("int32"), axis=[n_ax, h_ax, w_ax]),
            name="Output",
            tag="reduce_sum_compute",
        )

    if data_layout == "NHWC":
        output = te.compute(
            (batch_size, weight_height, weight_width, channels),
            lambda n, h, w, c: te.sum(data[n+n_ax, h+stride_h*h_ax , w+stride_w*w_ax, c].astype("int32"), axis=[n_ax, h_ax, w_ax]),
            name="Output",
            tag="reduce_sum_compute",
        )
    return output
