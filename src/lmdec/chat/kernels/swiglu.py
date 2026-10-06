from pathlib import Path

from torch import Tensor
from torch.utils.cpp_extension import load

_here = Path(__file__).parent

_ext = load(
    name="swiglu",
    sources=[str(_here / "swiglu.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


def swiglu(
    gate: Tensor,
    up: Tensor,
) -> Tensor:
    return _ext.swiglu_forward(gate, up)
