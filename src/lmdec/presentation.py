from lmdec.format import format_bytes, format_params
from lmdec.model_families.base import ModelAnalysis, ModelSpec, Workload
from lmdec.report import Row, Section, render_text


def render_analysis(analysis: ModelAnalysis, explain: bool = False) -> str:
    return render_text(build_report(analysis, explain))


def build_report(analysis: ModelAnalysis, explain: bool = False) -> list[Section]:
    spec = analysis.spec
    workload = analysis.workload
    memory = analysis.memory

    sections = [
        Section(spec.model_id, rule=True),
        Section(
            "MODEL",
            [
                Row("Architecture", spec.architecture),
                Row("Parameters", f"~{format_params(analysis.params.total)}"),
                Row("Layers", f"{spec.num_layers:,}"),
                Row("Hidden size", f"{spec.hidden_size:,}"),
                Row("Attention heads", f"{spec.num_attention_heads:,}"),
                *spec.attention.dimension_rows(),
                Row("Max context", f"{spec.context_window:,}"),
            ],
        ),
        Section(
            "MEMORY",
            [
                Row(
                    f"Weights ({workload.dtype.label})",
                    f"~{format_bytes(memory.weight_bytes)}",
                ),
                Row(
                    f"KV cache ({workload.kv_dtype.label})",
                    format_bytes(memory.kv_bytes_total),
                ),
                Row("Total", f"~{format_bytes(memory.total_bytes)}"),
            ],
        ),
        Section(
            f"KV CACHE ({workload.kv_dtype.label})",
            [
                *spec.attention.cache_layout_rows(),
                Row("Per token", format_bytes(memory.kv_bytes_per_token)),
                Row("Context", f"{workload.context:,} tokens"),
                Row("Batch size", f"{workload.batch_size:,}"),
                Row("Total", format_bytes(memory.kv_bytes_total)),
            ],
        ),
        Section("ATTENTION", spec.attention.summary_rows()),
    ]

    if explain:
        sections.extend(_derivation_sections(analysis))

    return sections


def _derivation_sections(analysis: ModelAnalysis) -> list[Section]:
    spec = analysis.spec
    workload = analysis.workload
    memory = analysis.memory
    per_token = format_bytes(memory.kv_bytes_per_token)

    sections = [
        Section("DERIVATION", rule=True),
        Section(
            "PARAMETERS",
            [
                f"{spec.architecture} architecture estimate",
                f"→ {analysis.params.total:,} parameters",
                *_embedding_lines(spec),
            ],
        ),
        *spec.attention.derivation_sections(spec.hidden_size),
        Section(
            "WEIGHT MEMORY",
            [
                (
                    f"{analysis.params.total:,} parameters × "
                    f"{workload.dtype.bytes_per_element} bytes "
                    f"({workload.dtype.label})"
                ),
                f"→ {format_bytes(memory.weight_bytes)}",
            ],
        ),
        Section(
            "KV CACHE",
            [
                *spec.attention.cache_derivation_lines(
                    spec.num_layers, workload.kv_dtype
                ),
                f"→ {per_token}/token",
            ],
        ),
        Section(
            lines=[
                f"{per_token}/token × {workload.context:,} tokens",
                f"→ {format_bytes(memory.kv_bytes_per_sequence)}/sequence",
            ],
        ),
    ]

    if workload.batch_size > 1:
        sections.append(
            Section(
                lines=[
                    (
                        f"{per_token}/token × {workload.context:,} tokens × "
                        f"{workload.batch_size:,} sequences"
                    ),
                    f"→ {format_bytes(memory.kv_bytes_total)} total KV cache",
                ],
            )
        )

    sections.append(
        Section(
            "ASSUMPTIONS",
            [
                "Weight memory is raw parameter storage only.",
                "KV memory is theoretical tensor payload.",
                *spec.attention.assumptions(),
                *_batch_lines(workload),
                "Activations, temporary workspaces, allocator/runtime overhead,",
                "cache block rounding, and runtime-specific layouts are excluded.",
            ],
            rule=True,
        )
    )

    return sections


def _embedding_lines(spec: ModelSpec) -> list[str]:
    matrix_size = f"{spec.vocab_size:,} × {spec.hidden_size:,}"

    if spec.tie_word_embeddings:
        return [
            "Input and output embeddings are tied; the LM head adds no",
            f"separate {matrix_size} parameter matrix.",
        ]

    return [
        "Input and output embeddings are untied; the estimate includes a",
        f"separate {matrix_size} parameter matrix for the LM head.",
    ]


def _batch_lines(workload: Workload) -> list[str]:
    if workload.batch_size == 1:
        return [
            "Batch size 1 means one sequence occupying the full",
            f"{workload.context:,}-token context.",
        ]

    return [
        (
            f"Batch size {workload.batch_size:,} means "
            f"{workload.batch_size:,} sequences, each occupying the full"
        ),
        f"{workload.context:,}-token context.",
    ]
