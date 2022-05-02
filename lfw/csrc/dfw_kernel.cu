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
    const scalar_t *query,
    const scalar_t *key,
    const scalar_t *value,
    const scalar_t *state,
    scalar_t *final_state,
    scalar_t *outputs,
    int b_size, int l_size, int d_size, int m_size)
{
    // TODO: this is 0 for now
    int m = 0; // blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;
    int b = blockDim.z * blockIdx.z + threadIdx.z;

    // Holds state (for this specific dimension d)
    // Dynamic shared memory
    extern __shared__ char smem[];
    scalar_t *shared_kv = reinterpret_cast<scalar_t *>(smem);

    // Check bounds
    // TODO: Check m bounds
    if (b < b_size && d < d_size)
    {
        // b, d, m
        int state_offset = (b * d_size + d) * m_size + m;
        // b, t, d, where t = 0
        int d_offset = (b * l_size) * d_size + d;
        // b, t, m, where t = 0
        int k_offset = (b * l_size) * m_size + m;

        // scalar_t cur_s = state[state_offset];
        // Load current state
        for (int m_local = 0; m_local < m_size; m_local++)
        {
            shared_kv[m_local] = state[state_offset + m_local];
        }

        // Go over each time step
        for (int t = 0; t < l_size; t++)
        {
            // Compute next state
            scalar_t out = 0;
            for (int m_local = 0; m_local < m_size; m_local++)
            {
                // Add new value to state
                shared_kv[m_local] += value[d_offset] * key[k_offset + m_local];
                // Query the state
                out += shared_kv[m_local] * query[k_offset + m_local];
            }

            // atomicAdd(
            //     &outputs[d_offset],
            //     out);
            // TODO: Won't parallelize
            outputs[d_offset] = out;

            d_offset += d_size;
            k_offset += m_size;
        }

        // Store final state

        // Load current state
        for (int m_local = 0; m_local < m_size; m_local++)
        {
            final_state[state_offset + m_local] = shared_kv[m_local];
        }
    }
}

// template <typename scalar_t>
// __global__ void lfw_cuda_bwd_kernel(
//     const scalar_t *grad_output, const scalar_t *grad_state,
//     const scalar_t *f, const scalar_t *query,
//     const scalar_t *f_key, const scalar_t *outputs,
//     scalar_t *s_grad, scalar_t *d_state,
//     int b_size, int l_size, int d_size, int k_size)
// {
//     int k = blockDim.x * blockIdx.x + threadIdx.x;
//     int d = blockDim.y * blockIdx.y + threadIdx.y;
//     int b = blockDim.z * blockIdx.z + threadIdx.z;

//     // Check bounds
//     if (b < b_size && d < d_size && k < k_size)
//     {
//         // b, l, d, k
//         const int maxT = l_size - 1;
//         const int state_flat_offset = (b * d_size + d) * k_size + k;
//         int state_offset = ((b * l_size + maxT) * d_size + d) * k_size + k;
//         // b, t, d, where t = max
//         int d_offset = (b * l_size + maxT) * d_size + d;
//         // b, t, k, where t = max
//         int k_offset = (b * l_size + maxT) * k_size + k;

//         auto cur_s_grad = grad_state[state_flat_offset];

//         for (int t = 0; t < l_size; t++)
//         {
//             cur_s_grad = grad_output[d_offset] * query[k_offset] + cur_s_grad;
//             s_grad[state_offset] = cur_s_grad;

//             // Compute next state
//             // Apply current f to gradient
//             cur_s_grad *= f[d_offset] * f_key[k_offset];

//             state_offset -= d_size * k_size;
//             d_offset -= d_size;
//             k_offset -= k_size;
//         }
//         d_state[state_flat_offset] = cur_s_grad;
//     }
// }

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
std::vector<torch::Tensor> lfw_cuda_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor state)
{
    // Length
    const auto L = value.size(1);
    // Batch
    const auto B = state.size(0);
    // Dimension
    const auto D = state.size(1);
    // Expansion dimension
    const auto M = state.size(2);

    // TODO: to save memory could we write back into the same state?
    auto final_state = torch::empty(
        {B, D, M},
        torch::TensorOptions()
            .dtype(value.dtype())
            .device(value.device()));

    auto outputs = torch::empty(
        {B, L, D},
        torch::TensorOptions()
            .dtype(value.dtype())
            .device(value.device()));

    const auto numMThreads = 1;
    // std::min(nextPowerOf2(K), MAX_K_TPB);
    const auto numDThreads = 1;
    // std::min(
    //     nextPowerOf2(D),
    //     std::min(MAX_TPB / numMThreads, MAX_D_TPB));

    const dim3 threads(numMThreads, numDThreads, 1);
    const dim3 blocks(
        1,
        // ceil_div(M, threads.x),
        ceil_div(D, threads.y),
        B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        value.scalar_type(),
        "lfw_cuda_fwd_kernel",
        ([&]
         { lfw_cuda_fwd_kernel<scalar_t><<<blocks, threads, M * sizeof(scalar_t)>>>(
               query.data<scalar_t>(),
               key.data<scalar_t>(),
               value.data<scalar_t>(),
               state.data<scalar_t>(),
               final_state.data<scalar_t>(),
               outputs.data<scalar_t>(),
               B, L, D, M); }));

    return {outputs, final_state};
}

// std::vector<torch::Tensor> lfw_cuda_backward(
//     torch::Tensor grad_output,
//     torch::Tensor grad_state,
//     torch::Tensor query,
//     torch::Tensor outputs)
// {
//     // Length
//     const auto L = query.size(1);
//     // Batch
//     const auto B = grad_state.size(0);
//     // Dimension
//     const auto D = grad_state.size(1);
//     // Expansion dimension
//     const auto K = grad_state.size(2);

//     // Kernel outputs
//     auto s_grad = torch::empty(
//         {B, L, D, K},
//         torch::TensorOptions()
//             .dtype(grad_state.dtype())
//             .device(grad_state.device()));
//     auto d_state = torch::empty(
//         {B, D, K},
//         torch::TensorOptions()
//             .dtype(grad_state.dtype())
//             .device(grad_state.device()));

//     const auto numMThreads = std::min(nextPowerOf2(K), MAX_K_TPB);
//     const auto numDThreads = std::min(
//         nextPowerOf2(D),
//         std::min(MAX_TPB / numMThreads, MAX_D_TPB));

//     const dim3 threads(numMThreads, numDThreads, 1);
//     const dim3 blocks(
//         ceil_div(K, threads.x),
//         ceil_div(D, threads.y),
//         B);

//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//         grad_state.scalar_type(),
//         "lfw_cuda_bwd_kernel",
//         ([&]
//          { lfw_cuda_bwd_kernel<scalar_t><<<blocks, threads>>>(
//                grad_output.data<scalar_t>(),
//                grad_state.data<scalar_t>(),
//                query.data<scalar_t>(),
//                outputs.data<scalar_t>(),
//                s_grad.data<scalar_t>(),
//                d_state.data<scalar_t>(),
//                B, L, D, K); }));
//     return {s_grad, d_state};
// }