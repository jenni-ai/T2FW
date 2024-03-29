#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// TODO: These constants may be device-dependent
// Maximum threads per block
const unsigned int MAX_TPB = 1024;
// Maximum thread for 1st dim of block
const unsigned int MAX_K_TPB = 1024;
// Maximum thread for 2nd dim of block
const unsigned int MAX_D_TPB = 1024;

#define ceil_div(a, b) (((a) + (b)-1) / (b))

template <typename scalar_t>
__global__ void lfw_cuda_fwd_kernel(
    const scalar_t *x, const scalar_t *f, const scalar_t *key,
    const scalar_t *f_key, const scalar_t *state, scalar_t *ckpt_states,
    int b_size, int l_size, int d_size, int k_size)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;
    int b = blockDim.z * blockIdx.z + threadIdx.z;

    // Check bounds
    if (b < b_size && d < d_size && k < k_size)
    {
        // b, d, k
        int state_offset = (b * d_size + d) * k_size + k;
        // b, t, d, where t = 0
        int d_offset = (b * l_size) * d_size + d;
        // b, t, k, where t = 0
        int k_offset = (b * l_size) * k_size + k;

        scalar_t cur_s = state[state_offset];

        // b, l, d, k
        state_offset = ((b * l_size) * d_size + d) * k_size + k;

        for (int t = 0; t < l_size; t++)
        {
            // Compute next state
            scalar_t f_full = f[d_offset] * f_key[k_offset];
            scalar_t x_full = x[d_offset] * key[k_offset];

            // Compute next state (TODO: FP32 accumulation?)
            cur_s = cur_s * f_full + x_full;

            ckpt_states[state_offset] = cur_s;

            state_offset += d_size * k_size;
            d_offset += d_size;
            k_offset += k_size;
        }
    }
}

template <typename scalar_t>
__global__ void lfw_cuda_bwd_kernel(
    const scalar_t *grad_output, const scalar_t *grad_state,
    const scalar_t *f, const scalar_t *query,
    const scalar_t *f_key, const scalar_t *outputs,
    scalar_t *s_grad, scalar_t *d_state,
    int b_size, int l_size, int d_size, int k_size)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;
    int b = blockDim.z * blockIdx.z + threadIdx.z;

    // Check bounds
    if (b < b_size && d < d_size && k < k_size)
    {
        // b, l, d, k
        const int maxT = l_size - 1;
        const int state_flat_offset = (b * d_size + d) * k_size + k;
        int state_offset = ((b * l_size + maxT) * d_size + d) * k_size + k;
        // b, t, d, where t = max
        int d_offset = (b * l_size + maxT) * d_size + d;
        // b, t, k, where t = max
        int k_offset = (b * l_size + maxT) * k_size + k;

        auto cur_s_grad = grad_state[state_flat_offset];

        for (int t = 0; t < l_size; t++)
        {
            cur_s_grad = grad_output[d_offset] * query[k_offset] + cur_s_grad;
            s_grad[state_offset] = cur_s_grad;

            // Compute next state
            // Apply current f to gradient
            cur_s_grad *= f[d_offset] * f_key[k_offset];

            state_offset -= d_size * k_size;
            d_offset -= d_size;
            k_offset -= k_size;
        }
        d_state[state_flat_offset] = cur_s_grad;
    }
}

// Compute power of two greater than or equal to `n`
unsigned int nextPowerOf2(unsigned int n)
{
    unsigned count = 0;

    // First n in the below condition
    // is for the case where n is 0
    if (n && !(n & (n - 1)))
        return n;

    while (n != 0)
    {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}

// CUDA declarations
torch::Tensor lfw_cuda_forward(
    torch::Tensor x,
    torch::Tensor f,
    torch::Tensor key,
    torch::Tensor f_key,
    torch::Tensor state)
{
    // Length
    const auto L = x.size(1);
    // Batch
    const auto B = state.size(0);
    // Dimension
    const auto D = state.size(1);
    // Expansion dimension
    const auto K = state.size(2);

    auto ckpt_states = torch::empty(
        {B, L, D, K},
        torch::TensorOptions()
            .dtype(x.dtype())
            .device(x.device()));

    const auto numKThreads = std::min(nextPowerOf2(K), MAX_K_TPB);
    const auto numDThreads = std::min(
        nextPowerOf2(D),
        std::min(MAX_TPB / numKThreads, MAX_D_TPB));

    const dim3 threads(numKThreads, numDThreads, 1);
    const dim3 blocks(
        ceil_div(K, threads.x),
        ceil_div(D, threads.y),
        B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(),
        "lfw_cuda_fwd_kernel",
        ([&]
         { lfw_cuda_fwd_kernel<scalar_t><<<blocks, threads>>>(
               x.data<scalar_t>(),
               f.data<scalar_t>(),
               key.data<scalar_t>(),
               f_key.data<scalar_t>(),
               state.data<scalar_t>(),
               ckpt_states.data<scalar_t>(),
               B, L, D, K); }));

    return ckpt_states;
}

std::vector<torch::Tensor> lfw_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor f,
    torch::Tensor query,
    torch::Tensor f_key,
    torch::Tensor outputs)
{
    // Length
    const auto L = query.size(1);
    // Batch
    const auto B = grad_state.size(0);
    // Dimension
    const auto D = grad_state.size(1);
    // Expansion dimension
    const auto K = grad_state.size(2);

    // Kernel outputs
    auto s_grad = torch::empty(
        {B, L, D, K},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));
    auto d_state = torch::empty(
        {B, D, K},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));

    const auto numKThreads = std::min(nextPowerOf2(K), MAX_K_TPB);
    const auto numDThreads = std::min(
        nextPowerOf2(D),
        std::min(MAX_TPB / numKThreads, MAX_D_TPB));

    const dim3 threads(numKThreads, numDThreads, 1);
    const dim3 blocks(
        ceil_div(K, threads.x),
        ceil_div(D, threads.y),
        B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_state.scalar_type(),
        "lfw_cuda_bwd_kernel",
        ([&]
         { lfw_cuda_bwd_kernel<scalar_t><<<blocks, threads>>>(
               grad_output.data<scalar_t>(),
               grad_state.data<scalar_t>(),
               f.data<scalar_t>(),
               query.data<scalar_t>(),
               f_key.data<scalar_t>(),
               outputs.data<scalar_t>(),
               s_grad.data<scalar_t>(),
               d_state.data<scalar_t>(),
               B, L, D, K); }));
    return {s_grad, d_state};
}