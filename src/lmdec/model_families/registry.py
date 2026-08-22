from transformers import PretrainedConfig

from .base import ModelFamily
from .generic_decoder import GenericDecoderFamily
from .qwen2 import Qwen2Family

FAMILIES: dict[str, ModelFamily] = {
    "qwen2": Qwen2Family(),
}

GENERIC_DECODER = GenericDecoderFamily()


def resolve_family(config: PretrainedConfig) -> ModelFamily:
    return FAMILIES.get(config.model_type, GENERIC_DECODER)
