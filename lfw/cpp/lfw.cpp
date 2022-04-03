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
/*
std::vector<torch::Tensor> lfw_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor f,
    torch::Tensor query,
    torch::Tensor f_key,
    torch::Tensor outputs,
    torch::Tensor s_grad,
    torch::Tensor d_state
);
*/
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
/*
std::vector<at::Tensor> lfw_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor f,
    torch::Tensor query,
    torch::Tensor f_key,
    torch::Tensor outputs,
    torch::Tensor s_grad,
    torch::Tensor d_state
) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(grad_state);
  CHECK_INPUT(f);
  CHECK_INPUT(query);
  CHECK_INPUT(f_key);
  CHECK_INPUT(outputs);
  CHECK_INPUT(s_grad);
  CHECK_INPUT(d_state);
  
  return lfw_cuda_backward(
    grad_output,
    grad_state,
    f,
    query,
    f_key,
    outputs,
    s_grad,
    d_state
  );
}
*/
TORCH_LIBRARY(lfw, m) {
  m.def("forward", lfw_forward);
}