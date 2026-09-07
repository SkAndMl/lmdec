from collections.abc import Iterable, Sequence
from dataclasses import dataclass

LABEL_WIDTH = 20
RULE_WIDTH = 40


@dataclass(frozen=True)
class Row:
    label: str
    value: str


Line = Row | str


@dataclass(frozen=True)
class Section:
    title: str | None = None
    lines: Sequence[Line] = ()
    rule: bool = False


def render_text(
    sections: Iterable[Section],
    label_width: int = LABEL_WIDTH,
    rule_width: int = RULE_WIDTH,
) -> str:
    return "\n\n".join(
        _render_section(section, label_width, rule_width) for section in sections
    )


def _render_section(section: Section, label_width: int, rule_width: int) -> str:
    lines: list[str] = []

    if section.title is not None:
        lines.append(section.title)
    if section.rule:
        lines.append("─" * rule_width)

    for line in section.lines:
        if isinstance(line, Row):
            lines.append(f"{line.label:<{label_width}}{line.value}")
        else:
            lines.append(line)

    return "\n".join(lines)
