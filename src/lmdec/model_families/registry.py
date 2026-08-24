from collections.abc import Callable

from transformers import PretrainedConfig

from .base import ModelFamily
from .generic_decoder import GenericDecoderFamily
from .qwen2 import Qwen2Family

ModelFamilyFactory = Callable[[str, PretrainedConfig], ModelFamily]

FAMILIES: dict[str, ModelFamilyFactory] = {
    "qwen2": Qwen2Family,
}

GENERIC_DECODER: ModelFamilyFactory = GenericDecoderFamily


def resolve_family(model_id: str, config: PretrainedConfig) -> ModelFamily:
    family_class = FAMILIES.get(config.model_type, GENERIC_DECODER)
    return family_class(model_id, config)
