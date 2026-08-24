from transformers import AutoConfig

from lmdec.model_families import resolve_family
from lmdec.presentation import render_analysis


def analyze(model_id: str) -> None:
    config = AutoConfig.from_pretrained(model_id)
    family = resolve_family(model_id, config)
    analysis = family.analyze()
    print(render_analysis(analysis))
