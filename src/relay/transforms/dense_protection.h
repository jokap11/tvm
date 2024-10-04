/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * Changed by Kappes Johannes @2024
 * TU Vienna ECS
 */

/*!
 *
 * \file fc_checksum_extension copy.cc
 *
 * \brief Extend each fully connected dunction with checksum generation (Huang et al.+ Sharif et Al)
 */
#include <tvm/ir/expr.h>
#include <tvm/tir/data_layout.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <tuple>

#include "pattern_utils.h"


/* Description of dense_checksum_extension
 *
 * The purpose of this pass is to find Fully Connected layer operators, which use only integers as in/output. These Operators are then extended 
 * with multiple Checksum Calculations after Huang et. Al. and Uzair. et. Al. to find soft faults during computation and return each Checksum result for each dense operator
 *
 *  x       w
 *   \     /
 *    \   /
 *    dense
 *      |
 *      |
 *      y
 *
 * and convert them into subgraphs with actual integer operations on x and w. Only interesting side fact is that the dense operator 
 * implicitly transposes the second input. Consequently leading to different axis usage for v,w= 1 vectors, which equaly row/coloumwise summation. 
 *
 * For Readability reasons x,w is copied
 * The used Algorithm is showed here:
 * 
 * Equation 1: (v^T * A)B ?= v^T C
 *
 *    x  w                   x                    w
 *    |  |                   |                    |
 *    |  |                   |                    |
 *    |  |                cast(32bit)             /
 *    |  |                   |                   /
 *    |  |           sum(axis=[0], keepdims)    /
 *    dense                     \              /
 *      |  \                     \            /
 *      |   \                dense(out_dtype="int32")
 *      | cast("int32")                /
 *      |      \                      /
 *      |   sum(axis=[0], keepdims)  /
 *      |              \            /
 *      |               \          /
 *      |                not_equal
 *      |                     |
 *      |                     |
 *      |                  cast(int8)
 *      |                     |
 *      |                     |
 *      |                   sum()
 *      |                     |
 *      |                     |
 *      y                 Output bit1
 * 
 * Equation 2: A(Bw) ?= Cw
 *
 *    x  w                   x                    w
 *    |  |                   |                    |
 *    |  |                   |                    |
 *    |  |                   |                cast(32bit)
 *    |  |                   |                   /
 *    |  |                    \          sum(axis=[0], keepdims)
 *    dense                    \              /
 *      |  \                    \            /
 *      |   \                dense(out_dtype="int32")
 *      | cast("int32")                /
 *      |      \                      /
 *      |   sum(axis=[1], keepdims)  /
 *      |              \            /
 *      |               \          /
 *      |                not_equal
 *      |                     |
 *      |                     |
 *      |                  cast(int8)
 *      |                     |
 *      |                     |
 *      |                   sum()
 *      |                     |
 *      |                     |
 *      y                 Output bit2
 * 
 * 
  %0 = cast(%data, dtype="int32")  ty=Tensor[(2, 4), int32] ;
  %1 = sum(%0, axis=[0], keepdims=True)  ty=Tensor[(1, 4), int32] ;
  %2 = nn.dense(%data, %kernel, units=None, out_dtype="int32")  ty=Tensor[(2, 4), int32] ;
  %3 = cast(%2, dtype="int32")  ty=Tensor[(2, 4), int32] ;
  %4 = nn.dense(%1, %kernel, units=None, out_dtype="int32")  ty=Tensor[(1, 4), int32] ;
  %5 = sum(%3, axis=[0], keepdims=True)  ty=Tensor[(1, 4), int32] ;
  %6 = not_equal(%4, %5)  ty=Tensor[(1, 4), bool] ;
  %7 = cast(%6, dtype="int8")  ty=Tensor[(1, 4), int8] ;
  %8 = cast(%kernel, dtype="int32")  ty=Tensor[(4, 4), int32] ;
  %9 = sum(%8, axis=[0], keepdims=True)  ty=Tensor[(1, 4), int32] ;
  %10 = nn.dense(%data, %9, units=None, out_dtype="int32")  ty=Tensor[(2, 1), int32] ;
  %11 = sum(%3, axis=[1], keepdims=True)  ty=Tensor[(2, 1), int32] ;
  %12 = not_equal(%10, %11)  ty=Tensor[(2, 1), bool] ;
  %13 = cast(%12, dtype="int8")  ty=Tensor[(2, 1), int8] ;
  %14 = sum(%7)  ty=int8 ;
  %15 = sum(%13)  ty=int8;
  (%2, %14, %15)  ty=(Tensor[(2, 4), int32], int8, int8)
*/

namespace tvm {
namespace relay {

// Find dense operations in dataflowgraph and bring them into an array
class DenseVisitor : private ExprVisitor {
 public:
  DenseVisitor() : dense_op(Op::Get("nn.dense")) {}

   Array<ObjectRef> Search(const Expr& expr) {
    VisitExpr(expr);
    return memo_;
  }

 private:
  //only supports 8bit dtypes as 16 bit is too big to be supported for accurate checksums
  bool supportedInputTensorType(const TensorTypeNode* arg){
    return ((arg->dtype == DataType::Int(8))  || (arg->dtype == DataType::UInt(8)));
  }


  void VisitExpr_(const CallNode* n) final {
    if (n->op == dense_op) {
      // detect (u)int8/16 * (u)int8/16 -> (u)int32/64 FC layers
      auto attr = n->attrs.as<DenseAttrs>();
      auto input  = n->args[0]->type_as<TensorTypeNode>();
      auto weight = n->args[1]->type_as<TensorTypeNode>();
      ICHECK(attr);
      ICHECK(input);
      ICHECK(weight);
      if(supportedInputTensorType(input)){
        memo_.push_back(GetRef<Call>(n));
      }
   }
    // iterate deeper levels
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }
  const Op& dense_op;
  Array<ObjectRef> memo_;            // Array for all already existing FC operation
};

Array<ObjectRef> SearchDense(const Expr& e) { return DenseVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_dense").set_body_typed(SearchDense);

// We dont want to exchange single nodes in the graph => No Mutation



Array<Call> mvp_method(const Expr& origin_expr){



    //Save ops reference to reduce access time
    const Op& cast_op = Op::Get("cast");
    const Op& sum_op = Op::Get("sum");
    const Op& neq_op = Op::Get("not_equal");
    const Op& dense_op = Op::Get("nn.dense");



    ///STATIC ATTRIBUTES
    //Cast Attr
    //used to cast bool to aot compatible memory planner
    auto cast_attr_8bit = make_object<CastAttrs>();
    cast_attr_8bit->dtype = DataType::Int(8);
    //32bit
    auto cast_attr_32bit = make_object<CastAttrs>();
    cast_attr_32bit->dtype = DataType::Int(32);
    //64bit
    auto cast_attr_64bit = make_object<CastAttrs>();
    cast_attr_64bit->dtype = DataType::Int(64);
    //elemwise sum operation
    auto reduce_elemwise_attrs = make_object<ReduceAttrs>();
    reduce_elemwise_attrs->axis = {0,1};  // 2D -> 0d
    reduce_elemwise_attrs->keepdims = false;  // 2D -> 0d
    reduce_elemwise_attrs->exclude  = false;
    //coloumnwise summation for (y,x) layout
    auto reduce_coloumnwise_attrs = make_object<ReduceAttrs>();
    reduce_coloumnwise_attrs->axis = {0};
    reduce_coloumnwise_attrs->keepdims = true;
    reduce_coloumnwise_attrs->exclude  = false;
    //rowwise summation for (y,x) layout
    auto reduce_rowwise_attrs = make_object<ReduceAttrs>();
    reduce_rowwise_attrs->axis = {1};
    reduce_rowwise_attrs->keepdims = true;
    reduce_rowwise_attrs->exclude  = false;




    Call origin_dense = Downcast<Call>(origin_expr);
    const auto* input_tensor  = origin_dense->args[0]->type_as<TensorTypeNode>();
    const auto* weight_tensor = origin_dense->args[1]->type_as<TensorTypeNode>();

    ICHECK(input_tensor != nullptr);
    ICHECK(weight_tensor != nullptr);

    const auto* origin_dense_attr = origin_dense->attrs.as<DenseAttrs>();
    ICHECK(origin_dense_attr != nullptr);
    //search layout string for position of N,C,H,W in data layout

    const auto input  = origin_dense->args[0];
    const auto weight = origin_dense->args[1];


    /// Equation 1: (v^T * A)B ?= v^T C


    Call input_32bit(cast_op, {input}, Attrs{cast_attr_32bit});
    Call input_coloumnwise_sum(sum_op, {input_32bit}, Attrs{reduce_coloumnwise_attrs});   
    //dense operations for both lhs equation vector matrix multiplications
    auto dense_attrs = make_object<DenseAttrs>();
    dense_attrs->out_dtype =  DataType::Int(32);
    //dense_attrs->units = Downcast<IntImm>(weight_tensor->shape[0]);
    Call lhs_eq1(dense_op, {input_coloumnwise_sum, weight}, Attrs{dense_attrs});

    // v^T C
    Call fc_32bit(cast_op, {origin_dense}, Attrs{cast_attr_32bit});
    Call fc_coloumnwise_sum(sum_op, {fc_32bit}, Attrs{reduce_coloumnwise_attrs});

    // not equal comparison of 2 vectors
    Call comp_eq1(neq_op, {lhs_eq1, fc_coloumnwise_sum});
    Call comp_eq1_8bit(cast_op, {comp_eq1}, Attrs{cast_attr_8bit});
    Call comp_eq1_sum(sum_op, {comp_eq1_8bit}, Attrs{reduce_elemwise_attrs});



    /// Equation 2: A(Bw) ?= Cw


    // A(Bw)
    Call weight_32bit(cast_op, {weight}, Attrs{cast_attr_32bit});
    Call weight_rowwise_sum(sum_op, {weight_32bit}, Attrs{reduce_coloumnwise_attrs}); // weight gets transposed in dense operation
    //dense operations for both lhs equation vector matrix multiplications
    auto dense_attrs2 = make_object<DenseAttrs>();
    dense_attrs2->out_dtype =  DataType::Int(32);
    //dense_attrs2->units = Downcast<IntImm>(input_tensor->shape[0]);
    Call lhs_eq2(dense_op, {input, weight_rowwise_sum}, Attrs{dense_attrs2});

    // Cw
    Call fc_rowwise_sum(sum_op, {fc_32bit}, Attrs{reduce_rowwise_attrs});

    // not equal comparison of 2 vectors
    Call comp_eq2(neq_op, {lhs_eq2, fc_rowwise_sum});
    Call comp_eq2_8bit(cast_op, {comp_eq2}, Attrs{cast_attr_8bit});
    Call comp_eq2_sum(sum_op, {comp_eq2_8bit}, Attrs{reduce_elemwise_attrs});

    return {comp_eq1_sum, comp_eq2_sum};
    }


}  // namespace relay
}  // namespace tvm
