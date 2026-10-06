#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void swiglu_kernel(
    const __half* gate,
    const __half* up,
    __half* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);

    float silu = g / (1.0f + expf(-g));
    out[i] = __float2half_rn(silu * u);
}

torch::Tensor swiglu_forward(
    torch::Tensor gate,
    torch::Tensor up
) {
    TORCH_CHECK(
        gate.is_cuda(),
        "gate must be CUDA"
    );

    TORCH_CHECK(
        up.is_cuda(),
        "up must be CUDA"
    )

    TORCH_CHECK(
        gate.scalar_type() == torch::kFloat16,
        "gate must be float16"
    );

    TORCH_CHECK(
        up.scalar_type() == torch::kFloat16,
        "up must be float16"
    );

    TORCH_CHECK(
        gate.sizes() == up.sizes(),
        "gate and up must have same shape"
    );

    TORCH_CHECK(
        gate.is_contiguous() && up.is_contiguous(),
        "inputs must be contiguous"
    );

    auto out = torch::empty_like(gate);

    int n = gate.numel();

    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto stream = c10::cuda::getCurrentCUDAStream(
        gate.get_device()
    );

    swiglu_kernel<<<
        blocks,
        threads,
        0,
        stream.stream()
    >>>(
        reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "swiglu_forward",
        &swiglu_forward,
        "SwiGLU forward"
    );
}