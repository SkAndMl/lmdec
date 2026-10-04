from pathlib import Path

from torch import Tensor
from torch.utils.cpp_extension import load

_here = Path(__file__).parent

_ext = load(
    name="rmsnorm",
    sources=[str(_here / "rmsnorm.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


def rmsnorm(
    x: Tensor,
    weight: Tensor,
    eps: float,
) -> Tensor:
    return _ext.rmsnorm_forward(
        x,
        weight,
        eps,
    )
