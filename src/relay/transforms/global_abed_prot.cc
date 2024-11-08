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
 * \file global_abed_prot.cc
 *
 * \brief Combine FIC/MVP and DuplicateIsland method in one global Pass
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
#include <optional>

#include "pattern_utils.h"
#include "conv2D_protection.h"
#include "dense_protection.h"

/* Description of Abed Protection Pass:
 *
 * The purpose of this pass is to add redundancy into simple Feed Forward NNs to detect soft faults. This is done in a multi-level approach on the graph-level -> 
 * 1.) Frontend tries to detect different pattern thats are offloaded onto functions to create a new tuple elements. 
 * Possible functions are (needs to be extended to regard Fusion) which are applied in this priority level:
 * 2.a) FIC method for Conv2D https://ieeexplore.ieee.org/document/9366780 + depthwise Conv2D https://ieeexplore.ieee.org/document/10137207
 * 2.b) MVP for Dense https://ieeexplore.ieee.org/document/1676475 Weird implementation becuase of TVMs Transposing weight matrix
 * 2.c) Make redundancy Islands that is simply sequential DMR
 * 
 * 
 * 2c) Illustration
 *    input      
 *      |
 *     pad
 *      |
 *    dense
 *      | 
 *   bias_add
 *      | 
 *     relu
 *      | 
 *     clip
 *      | 
 *     cast
 *      | 
 *     clip
 *      | 
 *    conv2d
 *      | 
 *   bias_add
 *      |  
 *    softmax
 *      |
 *
 *
 * After Duplication
 * 
 *
 *                    input      
 *                      |
 *                     pad
 *                      |
 *                    dense
 *                     |    \
 *                     |    add
 *                     |      \
 *                     |       \
 *                  bias_add   bias_add
 *                    |          |
 *                   relu       relu 
 *                    |          |
 *                   clip       clip 
 *                    |          |
 *                   cast       cast
 *                    |          |
 *                   clip       clip 
 *                    |   \      |
 *                    |    \     |
 *                    |       not_equal
 *                  conv2d        |      
 *                /   |           |
 *             /      |           |
 *          /         |           |
 *       add          |           |
 *        |           |           |
 *     bias_add    bias_add       |
 *        |           |           |
 *        |           |           |
 *       softmax   softmax    cast(int8)
 *         \     /    |           |
 *       not_equal    |           |
 *          |         |           | 
 *       cast(int8)   |           |
 *          |         |           | 
 *         sum        |         sum()
 *        
 *        
 *        
*/

namespace tvm {
namespace relay {
namespace {


const Op& conv2d_op(Op::Get("nn.conv2d"));
const Op& dense_op(Op::Get("nn.dense"));
const Op& sum_op(Op::Get("sum"));
const Op& ones_op(Op::Get("ones"));
const Op& zeros_op(Op::Get("zeros"));
const Op& cast_op(Op::Get("cast"));
const Op& add_op(Op::Get("add"));
const Op& neq_op = Op::Get("not_equal");


//This is used as an Array to gather information how many nodes are in the next lower hierachy after a protected operation
Array<Call> still_to_visit;
  // Internal visiting counter
std::unordered_map<const ExprNode*, size_t> visit_counter_;

//determines if fork was hit (Required to switch between BFS AND DFS)


bool check_constant_creation_in_relayExpr(const Call node){
    return ( node->op == ones_op || node->op == zeros_op);
}


std::unordered_map<int, tvm::relay::TShapeDataDependent> sum_axes_map =
{
  { 1, {Integer(0)}},
  { 2, {Integer(0),Integer(1)}},
  { 3, {Integer(0),Integer(1),Integer(2)}},
  { 4, {Integer(0),Integer(1),Integer(2),Integer(3)}},
};


/// @brief simple Function to handle FIC method (now only a dummy)
/// @param last_node
/// @return 
Call protect_conv2d(const Call& last_node){
  for (auto node : last_node->args)
  {
    // skip constant nodes or first input
    if(node.as<VarNode>() || node.as<ConstantNode>() || node.as<GlobalVarNode>()){
      continue;
    }
    // multi-path graph and node already visited=duplicated=protected
    if (visit_counter_.find(node.get()) == visit_counter_.end())
    {
      Call last_call_node = Downcast<Call>(node);
      if(!check_constant_creation_in_relayExpr(last_call_node))
      {
        still_to_visit.push_back(last_call_node);
      }
    }
  }
  return fic_method(last_node);
};


/// @brief simple Function to handle MVP method (now only a dummy)
/// @param last_node
/// @return 
Array<Call> protect_dense(const Call& last_node){
  for (auto node : last_node->args)
  {
    // skip constant nodes or first input
    if(node.as<VarNode>() || node.as<ConstantNode>() || node.as<GlobalVarNode>()){
      continue;
    }
    // multi-path graph and node already visited=duplicated=protected
    if (visit_counter_.find(node.get()) == visit_counter_.end())
    {
      still_to_visit.push_back(Downcast<Call>(node));
    }
  }
  return mvp_method(last_node);
};


/// @brief returns Call Node with pushed in between dummy node (for already visited or protected nodes)
/// @param last_node Ref node to be connected via ADD
/// @return 
Call generate_dummy_ref(const Call& last_node){
  auto input_tensor  = last_node->type_as<TensorTypeNode>();
  //auto dummy_op = add_op;
  auto arg_1 = Zeros(input_tensor->shape, input_tensor->dtype);
  Call reconstruct_graph(add_op, {last_node, arg_1});
  return reconstruct_graph;
};



/// @brief LOW level Reconstruction function -> Duplicates underlying DAG until ConstExpr/VarExpr + protected Ops is detected (Later also pattern)
/// @param start  node to duplicate
/// @return reconstructed DAG tuple element connected with dummies to complex input nodes or simple copy for 
Expr duplicate_island(const Expr& last_node) {

  
  auto it = visit_counter_.find(last_node.get());
  
  // multi-path graph and node already visited=duplicated=protected
  if (it != visit_counter_.end())
  {
    return generate_dummy_ref(Downcast<Call>(last_node));
  }
  else
  {
    // What about ones/zeros???
    if(last_node.as<VarNode>() || last_node.as<ConstantNode>()  || last_node.as<GlobalVarNode>())
    {
      return last_node;
    }
    else
    {
      visit_counter_.insert({last_node.get(), 1});
      auto last_op = Downcast<Call>(last_node);
      if (last_op->op == dense_op || last_op->op == conv2d_op)
      {
        // simple Copy of ref with dummy (Shifted to here/Since only those nodes should be revisited)
        still_to_visit.push_back(last_op);
        return generate_dummy_ref(last_op);
      }
      else if(check_constant_creation_in_relayExpr(last_op))
      {
        // no dummy (since more or less created constant)
        return last_node;
      }
      else
      {
        Array<Expr> arg_list;
        for (auto arg : last_op->args)
        {
            arg_list.push_back(duplicate_island(arg));
        }
        Call copy_node(last_op->op, arg_list, Attrs(last_op->attrs));
        return copy_node;
      }
    }
  }
}

/// @brief Protect element with index ele_nr in still_to_visit array and append tuple element to output array
/// @param still_to_visit
/// @param ele_nr
/// @param output_array
void protect_element(size_t ele_nr, Array<Call>& output_array){
  if (still_to_visit[ele_nr]->op == dense_op)
  {
    for (auto mvp_eq : protect_dense(still_to_visit[ele_nr]))
    {
      output_array.push_back(mvp_eq);
    }
  }
  else if (still_to_visit[ele_nr]->op == conv2d_op)
  {
    output_array.push_back(protect_conv2d(still_to_visit[ele_nr]));
  }
  else
  {
    auto original_tensor  = still_to_visit[ele_nr]->type_as<TensorTypeNode>();
    //used to cast bool to aot compatible memory planner
    //Save ops reference to reduce access time
    if(!check_constant_creation_in_relayExpr(still_to_visit[ele_nr]))
    {
      auto cast_attr_8bit = make_object<CastAttrs>();
      cast_attr_8bit->dtype = DataType::Int(8);
      Call comp(neq_op, {still_to_visit[ele_nr],  duplicate_island(still_to_visit[ele_nr])}, Attrs());
      Call comp_8bit(cast_op, {comp}, Attrs{cast_attr_8bit});
      Expr comp_sum = MakeReduce(comp_8bit, sum_axes_map[original_tensor->shape.size()], false,  false, "sum");
      output_array.push_back(Downcast<Call>(comp_sum));
    }
  }
}


/// @brief Top level abed protection function -> Duplicates whole exp DAG from end to start while duplicating islands through recursive reconstruction
/// and detection of protected Operators. Each duplication island is apppended to an output tuple and connected via dummy nodes to avoid CSE for more 
/// complex expr than VarExpr/ConstExpr. Also to avoid multiple redundancy each node is only visited once and a side arm might be connected via a dummy ref
/// @param start  node to duplicate
/// @return reconstructed DAG tuple element connected with dummies to complex input nodes or simple copy for 
Array<Call> full_abed_protection(const Expr& top_node){
  
  // Global List of start nodes
  still_to_visit.push_back(Downcast<Call>(top_node));

  Array<Call> output_array;


  for (size_t start_node=0 ; start_node < still_to_visit.size() ; start_node++) {
      // put original ref into it

      //depending on node type -> call different generation func by observing pattern (protect_dense/protect_conv2d/duplicate_island)
    if (still_to_visit[start_node]->op == dense_op)
    {
      for (auto mvp_eq : protect_dense(still_to_visit[start_node]))
      {
        output_array.push_back(mvp_eq);
      }
    }
    else if (still_to_visit[start_node]->op == conv2d_op)
    {
      output_array.push_back(protect_conv2d(still_to_visit[start_node]));
    }
    else
    {
      protect_element(start_node, output_array);
    }
    // VLOG(2) << "Print out still_to_visit array size: \n"<< still_to_visit.size() << std::endl;
  }

return output_array;

};



} //namespace

// We dont want to exchange single nodes in the graph => No Mutation
namespace transform {


IRModule GlobalAbedProt(const IRModule& mod) {
  // required for Add function for module
  tvm::Map<GlobalVar, Function> updates;

  auto funcs = mod->functions;  // unorderd_map with global var(function name) and function body
  for (const auto& ele : funcs) {
    ICHECK_EQ(FreeVars(ele.second).size(), 0);
    if (const auto* n = ele.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      auto first_exp = func->body;
      Array<tvm::relay::Expr> output_expr;

      // get existing top level expression in DAG
      if (func->body.as<Tuple>()) {
        first_exp = Downcast<Tuple>(func->body);
      } else if (func->body.as<Call>()) {
        first_exp = Downcast<Call>(func->body);
      } else {
        ICHECK_EQ(1, 0) << "func->body should be either Call or Tuple node";
      }
      output_expr.push_back(first_exp);

      
      auto output_tuple_expr = full_abed_protection(Downcast<Call>(first_exp));
      for(uint ele_nr=0; ele_nr < output_tuple_expr.size() ; ele_nr++)
      {
        output_expr.push_back(output_tuple_expr[ele_nr]);
      }
      

      output_expr = Array<Expr>(output_expr.rbegin(), output_expr.rend());
      Tuple new_func_body(output_expr);

      VLOG(2) << "New functional agglomeration \n" << PrettyPrint(new_func_body);

      Array<Type> return_array;
      //last elem==original element
      TensorType comp_output({}, DataType::Int(8)); //boolean type has dim.size=0
      for(uint i=0; i < output_expr.size()-1; i++)
      {
        return_array.push_back(comp_output);
      }
      return_array.push_back(func->ret_type);
      TupleType final_ret_type(return_array);
      Function extended_func(func->params, new_func_body, final_ret_type, func->type_params);
      updates.Set(ele.first, Downcast<Function>(extended_func));

      // VLOG(1) << "Print out return type of new function: \n"
      //         << PrettyPrint(func->ret_type)
      //         << "and the function: \n"
      //         << PrettyPrint(extended_func) << std::endl;
    }
    // Use implemented function to update each global var/ func pair
    for (auto pair : updates) {
      mod->Add(pair.first, pair.second, true);
    }
  }
  return mod;
}



Pass GlobalAbedProt() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [&](IRModule m, PassContext pc) { return GlobalAbedProt(m); };

  return CreateModulePass(pass_func, 0, "GlobalAbedProt", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.GlobalAbedProt").set_body_typed([](){return GlobalAbedProt();});

}  // namespace transform

}  // namespace relay
}  // namespace tvm
