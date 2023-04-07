// Copyright 2023 Bytedance Ltd. and/or its affiliates.
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
 */
#include "ths_op_f.h"
#include "bert_transformer_ext.h"

namespace bytetransformer {
namespace torch_ths {

using torch::Tensor;

std::tuple<Tensor, Tensor, Tensor> TransformerEncoder(int64_t head_num, int64_t head_size, Tensor qkv_kernel, Tensor qkv_bias,
                          Tensor attr_output_kernel, Tensor attr_output_bias,
                          Tensor attr_output_layernorm_gamma, Tensor attr_output_layernorm_beta,
                          Tensor inter_kernel, Tensor inter_bias, Tensor output_kernel,
                          Tensor output_bias, Tensor output_layernorm_gamma,
                          Tensor output_layernorm_beta, Tensor input, Tensor attr_mask,
                          bool is_remove_padding, bool use_fused_attention) {
  const at::ScalarType _st = qkv_kernel.scalar_type();
  CHECK_INPUT(qkv_kernel, _st);                    // hidden_dim, hidden_dim * 3
  CHECK_INPUT(qkv_bias, _st);                      // hidden_dim * 3
  CHECK_INPUT(attr_output_kernel, _st);            // hidden_dim, hidden_dim
  CHECK_INPUT(attr_output_bias, _st);              // hidden_dim
  CHECK_INPUT_LOOSE(attr_output_layernorm_gamma);  // hidden_dim
  CHECK_INPUT_LOOSE(attr_output_layernorm_beta);   // hidden_dim
  CHECK_INPUT(inter_kernel, _st);                  // 4 * hidden_dim, hidden_dim
  CHECK_INPUT(inter_bias, _st);                    // 4 * hidden_dim
  CHECK_INPUT(output_kernel, _st);                 // hidden_dim, 4 * hidden_dim
  CHECK_INPUT(output_bias, _st);                   // hidden_dim
  CHECK_INPUT_LOOSE(output_layernorm_gamma);       // hidden_dim
  CHECK_INPUT_LOOSE(output_layernorm_beta);        // hidden_dim
  CHECK_INPUT(input, _st);
  CHECK_INPUT(attr_mask, _st);
  auto input_size = input.sizes();
  int batch_size = input_size[0];
  int seq_len = input_size[1];
  std::vector<Tensor> weights{qkv_kernel,
                              qkv_bias,
                              attr_output_kernel,
                              attr_output_bias,
                              attr_output_layernorm_gamma,
                              attr_output_layernorm_beta,
                              inter_kernel,
                              inter_bias,
                              output_kernel,
                              output_bias,
                              output_layernorm_gamma,
                              output_layernorm_beta};
  auto output = torch::empty_like(input);
  // cal_buf_size()
  torch_ext::IBTEncoder *btencoder = nullptr;
  switch (_st) {
    case at::ScalarType::Float:
      btencoder = new torch_ext::BTEncoder<float>(head_num, head_size, weights);
      break;
    case at::ScalarType::Half:
      btencoder = new torch_ext::BTEncoder<half>(head_num, head_size, weights);
      break;
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
  auto buf_tensor =
    torch::empty({(long int)btencoder->get_buf_size(batch_size, seq_len, is_remove_padding, use_fused_attention)}, torch::dtype(torch::kInt8).device(torch::kCUDA));
  void *buf = bytetransformer::torch_ext::get_ptr<void>(buf_tensor);
  btencoder->forward(batch_size, seq_len, input, attr_mask, output, buf, is_remove_padding,
                     use_fused_attention);
  delete btencoder;
  int input_tensor_size = batch_size * head_num * seq_len * head_size;

  Tensor k_cache;
  Tensor v_cache;
  switch (_st) {
    case at::ScalarType::Float:
      k_cache = torch::from_blob((float*)buf + 1 * input_tensor_size,input_tensor_size,torch::dtype(input.dtype()).device(torch::kCUDA));
      v_cache = torch::from_blob((float*)buf + 2 * input_tensor_size,input_tensor_size,torch::dtype(input.dtype()).device(torch::kCUDA));
      return {output, k_cache, v_cache};
      break;
    case at::ScalarType::Half:
      k_cache = torch::from_blob((half*)buf + 1 * input_tensor_size,input_tensor_size,torch::dtype(input.dtype()).device(torch::kCUDA));
      v_cache = torch::from_blob((half*)buf + 2 * input_tensor_size,input_tensor_size,torch::dtype(input.dtype()).device(torch::kCUDA));
      return {output, k_cache, v_cache};
      break;
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }

}

static auto registry = torch::RegisterOperators(
    "ByteTransformer::BertTransformer("
    "int head_num, int head_size,"
    "Tensor qkv_kernel, Tensor qkv_bias,"
    "Tensor attr_output_kernel, Tensor attr_output_bias,"
    "Tensor attr_output_layernorm_gamma, Tensor attr_output_layernorm_beta,"
    "Tensor inter_kernel, Tensor inter_bias, Tensor output_kernel, Tensor output_bias,"
    "Tensor output_layernorm_gamma, Tensor output_layernorm_beta, Tensor input, Tensor attr_mask,"
    "bool is_remove_padding = True, bool use_fused_attention = True) -> "
    "(Tensor,Tensor,Tensor)",
    &TransformerEncoder);

}  // namespace torch_ths
}  // namespace bytetransformer
