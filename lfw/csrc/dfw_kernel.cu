#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>

// TODO: These constants may be device-dependent
// Maximum threads per block
const unsigned int MAX_TPB = 1024;
// Maximum thread for 1st dim of block
const unsigned int MAX_X_TPB = MAX_TPB;

#define WARP_SIZE 32
#define ceil_div(a, b) (((a) + (b)-1) / (b))

template <uint size, typename scalar_t>
struct Vec
{
    scalar_t vec[size];

    //     auto &operator+=(const Vec<size, scalar_t> *other)
    //     {
    // #pragma unroll
    //         for (int i = 0; i < size; i++)
    //         {
    //             this.vec[i] += other.vec[i];
    //         }
    //         return *this;
    //     }
};

/**
 * @brief Performs reduction within one active warp.
 */
template <typename scalar_t>
__device__ void warpReduce(volatile scalar_t *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

/**
 * @brief Performs parallel sum of an array, leaving the result in index 0.
 * Only supports EVEN sizes for correctness.
 *
 * @tparam scalar_t
 * @param arr
 * @param id
 * @param size
 * @return __device__
 */
template <uint width, typename VecType>
__device__ VecType sumReduc(
    VecType *arr,
    const int tid,
    const int size)
{
    for (unsigned int step_size = size / 2; step_size > WARP_SIZE; step_size >>= 1)
    {
        if (tid < step_size)
        {

#pragma unroll
            for (int i = 0; i < width; i++)
            {
                // Reduce to the left side
                arr[tid].vec[i] += arr[tid + step_size].vec[i];
            }
            // arr[tid] += arr[tid + step_size];
        }
        __syncthreads();
    }
    // auto res = arr[0];
    // if (tid < 32)
    //     warpReduce(arr, tid);

    //     VecType res;
    // #pragma unroll
    //     for (int w = 0; w < width; w++)
    //     {
    //         res.vec[w] = 0;
    //     }
    auto res = arr[0];
    for (int i = 1; i < min(WARP_SIZE, size); i++)
    {
#pragma unroll
        for (int w = 0; w < width; w++)
        {
            res.vec[w] += arr[i].vec[w];
        }
    }
    __syncthreads();
    return res;
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
    scalar_t *delta_value,
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
    // Holding tile results
    Vec<2, scalar_t> *shared_tile = reinterpret_cast<Vec<2, scalar_t> *>(&cur_state[m_size]);

    // NOTE: Shouldn't be possible to be out of bounds for (b and d)

    // We will be looping from m_start to m_end (which is the size of a tile)
    const int m_start = tile_id * tile_size;
    const int m_end = m_start + tile_size;

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
        shared_tile[tile_id].vec[0] = q_out;
        shared_tile[tile_id].vec[1] = k_out;
        __syncthreads();
        auto res = sumReduc<2>(shared_tile, tile_id, num_tiles);
        q_out = res.vec[0];
        k_out = res.vec[1];

        // Write output
        outputs[d_offset] = q_out;

        // Compute the delta
        auto curVal = value[d_offset] - k_out;
        delta_value[d_offset] = curVal;

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
 * @brief Computes the gradient of d_value, d_state.
 * Runs each dimension d in parallel.
 * Tiles along dimension m.
 */
template <typename scalar_t>
__global__ void lfw_cuda_bwd_value_kernel(
    const scalar_t *grad_output,
    const scalar_t *grad_state,
    const scalar_t *query,
    const scalar_t *key,
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
    // Holds recursive gradient of states (for this specific dimension d)
    scalar_t *cur_s_grad = reinterpret_cast<scalar_t *>(smem);
    // Holds tile results (size = num_tiles)
    Vec<1, scalar_t> *shared_tile = reinterpret_cast<Vec<1, scalar_t> *>(&cur_s_grad[d_size]);

    // NOTE: Shouldn't be possible to be out of bounds for (b and d)

    // We will be looping from m_start to m_end (which is the size of a tile)
    const int m_start = tile_id * tile_size;
    const int m_end = m_start + tile_size;

    const int maxT = l_size - 1;
    // b, d, m = 0
    const int state_offset = (b * d_size + d) * m_size;
    // b, t, d, where t = max, d=0
    int d_offset = (b * l_size + maxT) * d_size + d;
    // b, t, m, where t = max, m = 0
    int m_offset = (b * l_size + maxT) * m_size;

    for (int m = m_start; m < m_end && m < m_size; m++)
    {
        // Load final state's gradient
        cur_s_grad[m] = grad_state[state_offset + m];
    }

    // Loops from final timestep to first timestep
    for (int t = 0; t < l_size; t++)
    {
        scalar_t d_v = 0;
        for (int m = m_start; m < m_end && m < m_size; m++)
        {
            // Compute s_grad * k
            d_v += cur_s_grad[m] * key[m_offset + m];
        }

        shared_tile[tile_id].vec[0] = d_v;
        __syncthreads();
        auto res = sumReduc<1>(shared_tile, tile_id, num_tiles);
        d_v = res.vec[0];

        d_value[d_offset] = d_v;

        // Apply delta rule derivatives
        auto g_out = grad_output[d_offset];
        for (int m = m_start; m < m_end && m < m_size; m++)
        {
            auto change = g_out * query[m_offset + m] - d_v * key[m_offset + m];
            cur_s_grad[m] += change;
        }

        d_offset -= d_size;
        m_offset -= m_size;
    }

    // Store d_state
    for (int m = m_start; m < m_end && m < m_size; m++)
    {
        d_state[state_offset + m] = cur_s_grad[m];
    }
}

/**
 * @brief Computes the gradient of d_query and d_key.
 * Runs each dimension m in parallel.
 * Tiles along dimension d.
 */
template <typename scalar_t>
__global__ void lfw_cuda_bwd_qk_kernel(
    const scalar_t *grad_output,
    const scalar_t *grad_state,
    const scalar_t *query,
    const scalar_t *key,
    const scalar_t *delta_value,
    const scalar_t *final_state,
    scalar_t *d_query,
    scalar_t *d_key,
    const scalar_t *d_value,
    int b_size,
    int l_size,
    int d_size,
    int m_size,
    int num_tiles,
    int tile_size)
{
    const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int m = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockDim.z * blockIdx.z + threadIdx.z;

    // Dynamic shared memory
    extern __shared__ char smem[];
    // Holds state (for this specific dimension m)
    scalar_t *cur_state = reinterpret_cast<scalar_t *>(smem);
    // Holds d_state results
    scalar_t *cur_s_grad = &cur_state[d_size];
    // Holds tile results (size = num_tiles)
    Vec<2, scalar_t> *shared_tile = reinterpret_cast<Vec<2, scalar_t> *>(&cur_s_grad[d_size]);

    // NOTE: Shouldn't be possible to be out of bounds for (b and d)

    // We will be looping from m_start to m_end (which is the size of a tile)
    const int d_start = tile_id * tile_size;
    const int d_end = d_start + tile_size;
    const int width = 2;

    const int maxT = l_size - 1;
    // b, d, m, where d = 0
    const int state_offset = (b * d_size) * m_size + m;
    // b, t, d, where t = max, d=0
    int d_offset = (b * l_size + maxT) * d_size;
    // b, t, m, where t = max
    int m_offset = (b * l_size + maxT) * m_size + m;

    for (int d = d_start; d < d_end && d < d_size; d++)
    {
        // Load final state
        cur_state[d] = final_state[state_offset + d * m_size];
        // Load final state's gradient
        cur_s_grad[d] = grad_state[state_offset + d * m_size];
    }

    // Loops from final timestep to first timestep
    for (int t = 0; t < l_size; t++)
    {
        const auto q = query[m_offset];
        const auto k = key[m_offset];

        scalar_t d_q = 0;
        scalar_t d_k = 0;
        for (int d = d_start; d < d_end && d < d_size; d++)
        {
            auto deltaVal = delta_value[d_offset + d];
            // Move state backwards
            cur_state[d] -= deltaVal * k;

            d_q += grad_output[d_offset + d] * cur_state[d];

            // v * s_grad
            d_k += deltaVal * cur_s_grad[d];
            d_k -= d_value[d_offset + d] * cur_state[d];
        }
        shared_tile[tile_id].vec[0] = d_q;
        shared_tile[tile_id].vec[1] = d_k;
        __syncthreads();
        auto res = sumReduc<2>(shared_tile, tile_id, num_tiles);
        d_q = res.vec[0];
        d_k = res.vec[1];

        d_query[m_offset] = d_q;
        d_key[m_offset] = d_k;

        // Apply delta rule derivatives
        for (int d = d_start; d < d_end && d < d_size; d++)
        {
            auto change = grad_output[d_offset + d] * q - d_value[d_offset + d] * k;
            cur_s_grad[d] += change;
        }

        d_offset -= d_size;
        m_offset -= m_size;
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

    auto delta_value = torch::empty(
        {B, L, D},
        torch::TensorOptions()
            .dtype(value.dtype())
            .device(value.device()));

    // const auto max_tiles = std::min((uint)(std::sqrt(M) * 2), MAX_X_TPB);
    const auto num_tiles = nextPowerOf2(std::min((uint)(std::sqrt(M)), MAX_X_TPB));
    // const auto num_tiles = nextPowerOf2(std::min((uint)ceil_div(M, 2), MAX_X_TPB));
    // Elements to process per tile. Must be powers of 2.
    const auto tile_size = ceil_div(M, num_tiles);
    // std::min(
    //     nextPowerOf2(D),
    //     std::min(MAX_TPB / num_tiles, MAX_D_TPB));
    std::cout << "num_tiles: ";
    std::cout << num_tiles;
    std::cout << "\nTile size: ";
    std::cout << tile_size;
    std::cout << "\n";
    // Cannot use same sm for different dims
    const dim3 threads(num_tiles, 1, 1);
    const dim3 blocks(1, D, B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        value.scalar_type(),
        "lfw_cuda_fwd_kernel",
        ([&]
         { lfw_cuda_fwd_kernel<scalar_t><<<blocks, threads, M * sizeof(scalar_t) + (num_tiles) * sizeof(Vec<2, scalar_t>)>>>(
               query.data<scalar_t>(),
               key.data<scalar_t>(),
               value.data<scalar_t>(),
               state.data<scalar_t>(),
               final_state.data<scalar_t>(),
               outputs.data<scalar_t>(),
               delta_value.data<scalar_t>(),
               B, L, D, M, num_tiles, tile_size); }));

    return {outputs, final_state, delta_value};
}

std::vector<torch::Tensor> lfw_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_state,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor delta_value,
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
        {B, L, D},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));
    auto d_state = torch::empty(
        {B, D, M},
        torch::TensorOptions()
            .dtype(grad_state.dtype())
            .device(grad_state.device()));

    // TODO: Would be more efficient to pack rest of dimension into same SM
    auto num_tiles = nextPowerOf2(std::min((uint)(std::sqrt(M)), MAX_X_TPB));
    // Elements to process per tile
    auto tile_size = ceil_div(M, num_tiles);
    // Cannot use same sm for different dims
    dim3 threads(num_tiles, 1, 1);
    dim3 blocks(1, D, B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_state.scalar_type(),
        "lfw_cuda_bwd_value_kernel",
        ([&]
         { lfw_cuda_bwd_value_kernel<scalar_t><<<blocks, threads, M * sizeof(scalar_t) + num_tiles * sizeof(Vec<1, scalar_t>)>>>(
               grad_output.data<scalar_t>(),
               grad_state.data<scalar_t>(),
               query.data<scalar_t>(),
               key.data<scalar_t>(),
               // Outputs
               d_value.data<scalar_t>(),
               d_state.data<scalar_t>(),
               B, L, D, M, num_tiles, tile_size); }));

    num_tiles = nextPowerOf2(std::min((uint)(std::sqrt(D)), MAX_X_TPB));
    // Elements to process per tile
    tile_size = ceil_div(D, num_tiles);
    // Cannot use same sm for different dims
    dim3 threads_qk(num_tiles, 1, 1);
    dim3 blocks_qk(1, M, B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_state.scalar_type(),
        "lfw_cuda_bwd_qk_kernel",
        ([&]
         { lfw_cuda_bwd_qk_kernel<scalar_t><<<blocks_qk, threads_qk, (D * 2) * sizeof(scalar_t) + num_tiles * sizeof(Vec<2, scalar_t>)>>>(
               grad_output.data<scalar_t>(),
               grad_state.data<scalar_t>(),
               query.data<scalar_t>(),
               key.data<scalar_t>(),
               delta_value.data<scalar_t>(),
               final_state.data<scalar_t>(),
               // Outputs
               d_query.data<scalar_t>(),
               d_key.data<scalar_t>(),
               d_value.data<scalar_t>(),
               B, L, D, M, num_tiles, tile_size); }));

    return {d_query, d_key, d_value, d_state};
}