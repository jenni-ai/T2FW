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

  const auto ckpt_states = lfw_cuda_forward(x, f, key, f_key, state);
  // Computing output outside the kernel is ~20% faster
  const auto outputs = at::einsum("bldk,blk->bld", torch::TensorList({ckpt_states, query}));

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
  const auto d_state = res[1];

  const auto d_query = at::einsum("bld,bldk->blk", torch::TensorList({grad_output, ckpt_states}));
  const auto d_x = at::einsum("bldk,blk->bld", torch::TensorList({s_grad, key}));
  const auto d_key = at::einsum("bldk,bld->blk", torch::TensorList({s_grad, value}));

  // Multiply by state in place without using more memory
  s_grad.index({torch::indexing::Slice(), 0}) *= state;
  s_grad.index({torch::indexing::Slice(), torch::indexing::Slice(1)}) *= torch::slice(ckpt_states, 1, 0, -1);

  const auto d_f = at::einsum("bldk,blk->bld", torch::TensorList({s_grad, f_key}));
  const auto d_f_key = at::einsum("bldk,bld->blk", torch::TensorList({s_grad, forget}));

  return {d_x, d_f, d_query, d_key, d_f_key, d_state};
}

TORCH_LIBRARY(lfw, m) {
  m.def("forward", lfw_forward);
  m.def("backward", lfw_backward);
}