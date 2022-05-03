#include <torch/extension.h>

#include <vector>

// CUDA declarations
std::vector<torch::Tensor> lfw_cuda_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor state
);

std::vector<torch::Tensor> lfw_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor outputs,
    torch::Tensor final_state
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lfw_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor state
) {
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(value);
  CHECK_INPUT(state);

  return lfw_cuda_forward(query, key, value, state);
}

std::vector<at::Tensor> lfw_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor outputs,
    torch::Tensor final_state
) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(grad_state);
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(value);
  CHECK_INPUT(outputs);
  CHECK_INPUT(final_state);
  
  return lfw_cuda_backward(
    grad_output,
    grad_state,
    query,
    key,
    value,
    outputs,
    final_state
  );
}

TORCH_LIBRARY(dfw, m) {
  m.def("forward", lfw_forward);
  m.def("backward", lfw_backward);
}