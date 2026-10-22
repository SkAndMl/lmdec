import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def rmsnorm_kernel(
    X,
    W,
    Y,
    EPS: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * H + cols

    mask = cols < H

    x = tl.load(
        X + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    w = tl.load(
        W + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    square = x * x
    mean_square = tl.sum(square, axis=0) / H
    inv_rms = tl.rsqrt(mean_square + EPS)

    y = x * inv_rms * w

    tl.store(Y + offsets, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.device == w.device
    assert x.is_contiguous and w.is_contiguous
    assert x.dtype == w.dtype
    assert x.shape[-1] == w.numel()

    output = torch.empty_like(x)

    H = x.shape[-1]
    num_rows = x.numel() // H

    BLOCK_SIZE = triton.next_power_of_2(H)

    rmsnorm_kernel[(num_rows,)](
        X=x,
        W=w,
        Y=output,
        EPS=eps,
        H=H,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
