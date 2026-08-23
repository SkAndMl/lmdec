from transformers import PretrainedConfig

from .base import ModelFamily
from .generic_decoder import GenericDecoderFamily
from .qwen2 import Qwen2Family

FAMILIES: dict[str, ModelFamily] = {
    "qwen2": Qwen2Family,
}

GENERIC_DECODER = GenericDecoderFamily


def resolve_family(model_id: str, config: PretrainedConfig) -> ModelFamily:
    family_class = FAMILIES.get(config.model_type, GENERIC_DECODER)
    return family_class(model_id, config)
