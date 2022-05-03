#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// TODO: These constants may be device-dependent
// Maximum threads per block
const unsigned int MAX_TPB = 1024;
// Maximum thread for 1st dim of block
// TODO: Should be dynamically set based on m
const unsigned int MAX_TILES = 16;
// const unsigned int MAX_TILES = 1024;
// Maximum thread for 2nd dim of block
const unsigned int MAX_D_TPB = 1024;

#define ceil_div(a, b) (((a) + (b)-1) / (b))

/**
 * @brief Performs parallel sum of an array, leaving the result in index 0.
 *
 * @tparam scalar_t
 * @param arr
 * @param id
 * @param size
 * @return __device__
 */
template <typename scalar_t>
__device__ void parallel_sum(
    scalar_t *arr,
    const int id,
    const int size,
    const int width)
{
    int step_size = size / 2;
    while (step_size > 0)
    {
        if (id < step_size)
        {
#pragma unroll
            for (int w = 0; w < width; w++)
            {
                // Reduce to the left side
                auto curPos = id * width + w;
                arr[curPos] += arr[curPos + step_size * width];
            }
        }
        __syncthreads();
        step_size = step_size / 2;
    }
}

/**
 * @brief Runs each dimension d in parallel.
 */
template <typename scalar_t>
__global__ void lfw_cuda_fwd_kernel(
    const scalar_t *query,
    const scalar_t *key,
    const scalar_t *value,
    const scalar_t *state,
    scalar_t *final_state,
    scalar_t *outputs,
    int b_size,
    int l_size,
    int d_size,
    int m_size,
    int num_tiles,
    int tile_size)
{
    const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int d = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockDim.z * blockIdx.z + threadIdx.z;

    // Dynamic shared memory
    extern __shared__ char smem[];
    // Holds state (for this specific dimension d) (size = m_size)
    scalar_t *cur_state = reinterpret_cast<scalar_t *>(smem);
    // Holds tile results (size = num_tiles)
    scalar_t *shared_tile = &cur_state[m_size];

    // NOTE: Shouldn't be possible to be out of bounds for (b and d)

    // We will be looping from m_start to m_end (which is the size of a tile)
    const int m_start = tile_id * tile_size;
    const int m_end = m_start + tile_size;
    const int width = 2;

    // b, d, m, m = 0
    int state_offset = (b * d_size + d) * m_size;
    // b, t, d, where t = 0
    int d_offset = (b * l_size) * d_size + d;
    // b, t, m, where t = 0, m = 0
    int m_offset = (b * l_size) * m_size;

    // Load current state
    for (int m = m_start; m < m_end && m < m_size; m++)
    {
        cur_state[m] = state[state_offset + m];
    }

    // Go over each time step
    for (int t = 0; t < l_size; t++)
    {
        // TODO: Could load value in SM since it's reused?
        scalar_t q_out = 0;
        scalar_t k_out = 0;

        for (int m = m_start; m < m_end && m < m_size; m++)
        {
            // Query the state
            q_out += cur_state[m] * query[m_offset + m];
            // Query the old value associated with key
            k_out += cur_state[m] * key[m_offset + m];
        }
        // Each tile produce its partial results
        shared_tile[tile_id * width] = q_out;
        shared_tile[tile_id * width + 1] = k_out;
        __syncthreads();
        // Sum tile results via parallel reduction
        // parallel_sum(shared_tile, tile_id, num_tiles, width);
        // q_out = shared_tile[0];
        // k_out = shared_tile[1];

        // Non-parallel reduction seems to be faster
        q_out = 0;
        k_out = 0;
        for (int i = 0; i < num_tiles; i++)
        {
            q_out += shared_tile[i * width];
            k_out += shared_tile[i * width + 1];
        }
        __syncthreads();
        // Write output
        outputs[d_offset] = q_out;

        // Compute the delta
        auto curVal = value[d_offset] - k_out;

        // Compute next state
        // TODO: Could merge this loop?
        for (int m = m_start; m < m_end && m < m_size; m++)
        {
            // Add new value to state
            cur_state[m] += curVal * key[m_offset + m];
        }

        d_offset += d_size;
        m_offset += m_size;
    }

    // Store final state
    for (int m = m_start; m < m_end && m < m_size; m++)
    {
        final_state[state_offset + m] = cur_state[m];
    }
}

/**
 * @brief Computes the gradient of query and key.
 * Runs each dimension d in parallel.
 * Tiles along dimension m.
 */
template <typename scalar_t>
__global__ void lfw_cuda_bwd_kernel(
    const scalar_t *grad_output,
    const scalar_t *grad_state,
    const scalar_t *query,
    const scalar_t *key,
    const scalar_t *value,
    const scalar_t *outputs,
    const scalar_t *final_state,
    scalar_t *d_query,
    scalar_t *d_key,
    scalar_t *d_value,
    scalar_t *d_state,
    int b_size,
    int l_size,
    int d_size,
    int m_size,
    int num_tiles,
    int tile_size)
{
    const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int d = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockDim.z * blockIdx.z + threadIdx.z;

    // Dynamic shared memory
    extern __shared__ char smem[];
    // Holds state (for this specific dimension d)
    scalar_t *cur_state = reinterpret_cast<scalar_t *>(smem);
    // Holds d_state results
    scalar_t *cur_s_grad = &cur_state[d_size];
    // Holds tile results (size = num_tiles)
    scalar_t *shared_tile = &cur_s_grad[d_size];

    // NOTE: Shouldn't be possible to be out of bounds for (b and d)

    // We will be looping from m_start to m_end (which is the size of a tile)
    const int m_start = tile_id * tile_size;
    const int m_end = m_start + tile_size;

    const int maxT = l_size - 1;
    // b, d, m = 0
    const int state_offset = (b * d_size + d) * m_size;
    // b, t, d, where t = max, d=0
    // TODO
    int d_offset = (b * l_size + maxT) * d_size;
    //  + d;
    // b, t, m, where t = max, m = 0
    int m_offset = (b * l_size + maxT) * m_size;

    for (int m = m_start; m < m_end && m < m_size; m++)
    {
        // Load final state
        cur_state[m] = final_state[state_offset + m];
        // Load final state's gradient
        cur_s_grad[m] = grad_state[state_offset + m];
    }

    // Loops from final timestep to first timestep
    for (int t = 0; t < l_size; t++)
    {
        auto curVal = value[d_offset];
        scalar_t tmp_d_query = 0;
        scalar_t tmp_d_key = 0;

        for (int m = m_start; m < m_end && m < m_size; m++)
        {
            // Apply delta rule derivative (deriv of s_t w.r.t. s_{t-1})
            cur_s_grad[m] *= -key[m_offset + m] * key[m_offset + m];
        }

        for (int m = m_start; m < m_end && m < m_size; m++)
        {
            // Apply gradient from query
            cur_s_grad[m] += grad_output[d_offset] * query[m_offset + m];

            // d_query = grad_output * state
            // tmp_d_query += grad_output[d_offset] * cur_state[m];

            // Compute previous state (reversing)
            // cur_state[m] -= curVal * key[m_offset + m];

            // tmp_d_key += curVal * cur_state[m];
        }

        // Each tile produce its partial results
        // shared_tile[tile_id] = tmp_d_query;
        // __syncthreads();
        // Sum tile results via parallel reduction
        // d_query[m_offset] = parallel_sum(shared_tile, tile_id, num_tiles);

        // Each tile produce its partial results
        // shared_tile[tile_id] = tmp_d_key;
        // __syncthreads();
        // Sum tile results via parallel reduction
        // d_key[m_offset] = parallel_sum(shared_tile, tile_id, num_tiles);

        d_offset -= d_size;
        m_offset -= m_size;
    }

    for (int m = m_start; m < m_end && m < m_size; m++)
    {
        d_state[state_offset + m] = cur_s_grad[m];
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

    // TODO: Maybe optimize for tiles = tile_size?
    // TODO: Test with lower max k tpb
    // TODO: Would be more efficient to pack rest of dimension into same SM
    const auto num_tiles = std::min(nextPowerOf2(M), MAX_TILES);
    // Elements to process per tile
    const auto tile_size = ceil_div(M, num_tiles);
    // std::min(
    //     nextPowerOf2(D),
    //     std::min(MAX_TPB / num_tiles, MAX_D_TPB));

    // Cannot use same sm for different dims
    const dim3 threads(num_tiles, 1, 1);
    const dim3 blocks(
        1,
        // ceil_div(M, threads.x),
        ceil_div(D, threads.y),
        B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        value.scalar_type(),
        "lfw_cuda_fwd_kernel",
        ([&]
         { lfw_cuda_fwd_kernel<scalar_t><<<blocks, threads, (M + num_tiles * 2) * sizeof(scalar_t)>>>(
               query.data<scalar_t>(),
               key.data<scalar_t>(),
               value.data<scalar_t>(),
               state.data<scalar_t>(),
               final_state.data<scalar_t>(),
               outputs.data<scalar_t>(),
               B, L, D, M, num_tiles, tile_size); }));

    return {outputs, final_state};
}

std::vector<torch::Tensor> lfw_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor outputs,
    torch::Tensor final_state)
{
    // Length
    const auto L = query.size(1);
    // Batch
    const auto B = grad_state.size(0);
    // Dimension
    const auto D = grad_state.size(1);
    // Expansion dimension
    const auto M = grad_state.size(2);

    // Kernel outputs
    auto d_query = torch::empty(
        {B, L, M},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));
    auto d_key = torch::empty(
        {B, L, M},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));
    auto d_value = torch::empty(
        {B, L, M},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));
    auto d_state = torch::empty(
        {B, D, M},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));

    // TODO: Maybe optimize for tiles = tile_size?
    // TODO: Test with lower max k tpb
    // TODO: Would be more efficient to pack rest of dimension into same SM
    const auto num_tiles = std::min(nextPowerOf2(M), MAX_TILES);
    // Elements to process per tile
    const auto tile_size = ceil_div(M, num_tiles);
    // Cannot use same sm for different dims
    const dim3 threads(num_tiles, 1, 1);
    const dim3 blocks(
        1,
        ceil_div(D, threads.y),
        B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_state.scalar_type(),
        "lfw_cuda_bwd_kernel",
        ([&]
         { lfw_cuda_bwd_kernel<scalar_t><<<blocks, threads, (M * 2 + num_tiles) * sizeof(scalar_t)>>>(
               grad_output.data<scalar_t>(),
               grad_state.data<scalar_t>(),
               query.data<scalar_t>(),
               key.data<scalar_t>(),
               value.data<scalar_t>(),
               outputs.data<scalar_t>(),
               final_state.data<scalar_t>(),
               // Outputs
               d_query.data<scalar_t>(),
               d_key.data<scalar_t>(),
               d_value.data<scalar_t>(),
               d_state.data<scalar_t>(),
               B, L, D, M, num_tiles, tile_size); }));

    return {d_query, d_key, d_value, d_state};
}