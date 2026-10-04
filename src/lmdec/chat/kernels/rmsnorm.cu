#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void rmsnorm_kernel(
    const __half* x,
    const __half* weight,
    __half* out,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const int row_offset = row * hidden_size;

    float local_sum = 0.0f;

    for (int i=tid; i<hidden_size; i+=blockDim.x) {
        float v = __half2float(x[row_offset+i]);
        local_sum += v * v;
    }

    for (int offset=16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(
            0xffffffff,
            local_sum,
            offset
        );
    }

    const int lane = tid % 32;
    const int warp_id = tid / 32;

    __shared__ float warp_sums[8];

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        float value = lane < 8 ? warp_sums[lane] : 0.0f;

        for (int offset=16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(
                0xffffffff,
                value,
                offset
            );
        }

        if (lane == 0) {
            warp_sums[0] = value;
        }
    } 

    __syncthreads();

    const float inv_rms = rsqrtf(
        warp_sums[0] / hidden_size + eps
    );

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[row_offset + i]);
        float w = __half2float(weight[i]);

        out[row_offset + i] = __float2half_rn(v * inv_rms * w);
    }
}

torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    TORCH_CHECK(
        x.is_cuda(),
        "x must be CUDA"
    );

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16,
        "x must be float16"
    );

    TORCH_CHECK(
        weight.scalar_type() == torch::kFloat16,
        "weight must be float16"
    );

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous"
    );

    const int hidden_size = x.size(-1);

    TORCH_CHECK(
        weight.numel() == hidden_size,
        "weight size must equal hidden size"
    );

    const int rows = x.numel() / hidden_size;

    auto out = torch::empty_like(x);

    constexpr int threads = 256;

    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    rmsnorm_kernel<<<
        rows,
        threads,
        0,
        stream.stream()
    >>>(
        reinterpret_cast<const __half*>(
            x.data_ptr<at::Half>()
        ),
        reinterpret_cast<const __half*>(
            weight.data_ptr<at::Half>()
        ),
        reinterpret_cast<__half*>(
            out.data_ptr<at::Half>()
        ),
        hidden_size,
        static_cast<float>(eps)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rmsnorm_forward",
        &rmsnorm_forward,
        "RMSNorm forward"
    );
}