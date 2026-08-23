from transformers import AutoConfig

from lmdec.model_families import resolve_family

from .presentation import render_analysis


def analyze(model_id: str) -> None:
    config = AutoConfig.from_pretrained(model_id)
    family = resolve_family(model_id, config)
    total_params = family.estimate_params()
    kv_bytes_per_token = family.calculate_kv_cache_bytes(
        context_length=1,
        bytes_per_token=2,
    )

    print(render_analysis(family, total_params, kv_bytes_per_token))
