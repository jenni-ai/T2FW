#include <torch/extension.h>

#include <vector>

// CUDA declarations
torch::Tensor lfw_cuda_forward(
    torch::Tensor x,
    torch::Tensor f,
    torch::Tensor key,
    torch::Tensor f_key,
    torch::Tensor state
);

std::vector<torch::Tensor> lfw_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor f,
    torch::Tensor query,
    torch::Tensor f_key,
    torch::Tensor outputs
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lfw_forward(
    torch::Tensor x,
    torch::Tensor f,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor f_key,
    torch::Tensor state
) {
  CHECK_INPUT(x);
  CHECK_INPUT(f);
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(f_key);
  CHECK_INPUT(state);

  auto ckpt_states = lfw_cuda_forward(x, f, key, f_key, state);
  // Computing output outside the kernel is ~20% faster
  auto outputs = torch::squeeze(torch::matmul(ckpt_states, torch::unsqueeze(query, -1)), -1);

  return {outputs, ckpt_states};
}

std::vector<at::Tensor> lfw_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor value,
    torch::Tensor forget,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor f_key,
    torch::Tensor outputs,
    torch::Tensor state,
    torch::Tensor ckpt_states
) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(grad_state);
  CHECK_INPUT(value);
  CHECK_INPUT(forget);
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(f_key);
  CHECK_INPUT(outputs);
  CHECK_INPUT(state);
  CHECK_INPUT(ckpt_states);
  
  auto res = lfw_cuda_backward(
    grad_output,
    grad_state,
    forget,
    query,
    f_key,
    outputs
  );
  auto s_grad = res[0];
  auto d_state = res[1];

  // d_x [B, L, D, K] x [B, L, K, 1]
  auto d_x = torch::squeeze(torch::matmul(s_grad, torch::unsqueeze(key, -1)), -1);
  // d_query [B, L, 1, D] x [B, L, D, K]
  auto d_query = torch::squeeze(torch::matmul(torch::unsqueeze(grad_output, -2), ckpt_states), -2);
  // d_key [B, L, 1, D] x [B, L, D, K]
  auto d_key = torch::squeeze(torch::matmul(torch::unsqueeze(value, -2), s_grad), -2);

  // Multiply by state
  s_grad *= torch::cat(torch::TensorList({torch::unsqueeze(state, 1), torch::slice(ckpt_states, 1, 0, -1)}), 1);
  // d_f [B, L, D, K] x [B, L, K, 1]
  auto d_f = torch::squeeze(torch::matmul(s_grad, torch::unsqueeze(f_key, -1)), -1);
  // d_f_key [B, L, D, K] x [B, L, K, 1]
  auto d_f_key = torch::squeeze(torch::matmul(torch::unsqueeze(forget, -2), s_grad), -2);

  return {d_x, d_f, d_query, d_key, d_f_key, d_state};
}

TORCH_LIBRARY(lfw, m) {
  m.def("forward", lfw_forward);
  m.def("backward", lfw_backward);
}