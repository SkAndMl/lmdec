"""Generate every code-drawn diagram used by the regex-constrained decoding post.

The article keeps its three equations as native LaTeX in Markdown. This script
owns only the explanatory diagrams and benchmark visual.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent

# One restrained visual system for the full post.
INK = "#14213D"
MUTED = "#64748B"
BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
CYAN = "#0891B2"
CYAN_LIGHT = "#CFFAFE"
ORANGE = "#EA580C"
ORANGE_LIGHT = "#FFEDD5"
GREEN = "#15803D"
GREEN_LIGHT = "#DCFCE7"
RED = "#B91C1C"
RED_LIGHT = "#FEE2E2"
PAPER = "#FCFCFA"
PANEL = "#F8FAFC"
BORDER = "#D9E1EA"
WHITE = "#FFFFFF"


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "mathtext.fontset": "stix",
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def clean_canvas(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = BORDER,
    radius: float = 2.0,
    linewidth: float = 1.2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def label(ax: plt.Axes, x: float, y: float, text: str, *, color: str = MUTED) -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
        family="DejaVu Sans",
    )


def chip(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    text: str,
    *,
    face: str = WHITE,
    edge: str = BORDER,
    color: str = INK,
    alpha: float = 1.0,
    strike: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        8,
        boxstyle="round,pad=0.01,rounding_size=2",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + 4,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        alpha=alpha,
        family="DejaVu Sans Mono",
    )
    if strike:
        ax.plot([x + 1.3, x + width - 1.3], [y + 1.2, y + 6.8], color=RED, lw=1.6)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.5,
    curve: float = 0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def cover() -> None:
    fig, ax = clean_canvas((16, 8.6))

    # Faint construction grid makes the image feel technical without adding noise.
    for x in np.arange(5, 100, 5):
        ax.plot([x, x], [7, 93], color="#EEF2F6", lw=0.45, zorder=0)
    for y in np.arange(10, 95, 5):
        ax.plot([4, 96], [y, y], color="#EEF2F6", lw=0.45, zorder=0)

    label(ax, 7, 87, "01  MODEL LOGITS", color=BLUE)
    card(ax, 6, 21, 24, 61, face=WHITE)
    candidates = [
        ('"Jan"', 0.92, False),
        ('"2023"', 0.79, True),
        ('"the"', 0.64, False),
        ('"20"', 0.58, True),
        ('"-"', 0.31, False),
    ]
    for i, (token, score, valid) in enumerate(candidates):
        y = 69 - i * 10
        ax.text(9, y + 2.4, token, family="DejaVu Sans Mono", fontsize=10.5, va="center")
        ax.add_patch(Rectangle((16, y), 10 * score, 4.8, color=BLUE if valid else "#CBD5E1"))
        ax.text(27.5, y + 2.4, f"{score:.2f}", ha="right", va="center", fontsize=8.8, color=MUTED)

    label(ax, 38, 87, "02  REGEX STATE", color=ORANGE)
    card(ax, 36, 21, 28, 61, face=WHITE)
    ax.text(
        50,
        72,
        r"\d{4}-\d{2}-\d{2}",
        ha="center",
        va="center",
        family="DejaVu Sans Mono",
        fontsize=14,
        color=INK,
    )
    state_x = np.linspace(40, 60, 6)
    for i, x in enumerate(state_x):
        if i < len(state_x) - 1:
            arrow(ax, (x + 1.3, 52), (state_x[i + 1] - 1.3, 52), color=ORANGE, width=1.6)
        face = ORANGE_LIGHT if i == 2 else WHITE
        edge = ORANGE if i == 2 else BORDER
        ax.add_patch(Circle((x, 52), 1.65, facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(50, 43.5, "current state", ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    ax.text(50, 34.5, "valid token mask", ha="center", fontsize=9, color=MUTED)
    for i in range(12):
        x = 42.3 + (i % 6) * 3.1
        y = 27.5 + (i // 6) * 3.4
        ax.add_patch(
            Rectangle(
                (x, y),
                2.25,
                2.25,
                facecolor=CYAN if i in (1, 3, 6, 8) else "#E2E8F0",
                edgecolor="none",
            )
        )

    label(ax, 72, 87, "03  VALID OUTPUT", color=GREEN)
    card(ax, 70, 21, 24, 61, face=WHITE)
    output = list("2023-01-01")
    start_x = 72.1
    for i, char in enumerate(output):
        x = start_x + (i % 5) * 4.0
        y = 56 if i < 5 else 45
        face = GREEN_LIGHT if char != "-" else ORANGE_LIGHT
        edge = GREEN if char != "-" else ORANGE
        chip(ax, x, y, 3.2, char, face=face, edge=edge, color=INK)
    ax.text(82, 35, "fullmatch ✓", ha="center", color=GREEN, fontsize=11, fontweight="bold")

    arrow(ax, (30.5, 51.5), (35.5, 51.5), color=BLUE, width=2)
    arrow(ax, (64.5, 51.5), (69.5, 51.5), color=ORANGE, width=2)
    ax.text(50, 9, "MODEL PREFERENCE  ×  FORMAL CONSTRAINT", ha="center", fontsize=10,
            color=MUTED, fontweight="bold")
    save(fig, "00_cover.png")


def prompting_vs_constraints() -> None:
    fig, ax = clean_canvas((13.6, 5.4))
    label(ax, 5, 89, "SAME MODEL SCORES · DIFFERENT SUPPORT")

    # Two lanes share the exact same candidate ordering.
    lanes = [
        ("PROMPT ONLY", 65, False, 'selects "Jan"', RED, RED_LIGHT),
        ("REGEX MASK", 27, True, 'selects "2023"', GREEN, GREEN_LIGHT),
    ]
    candidate_tokens = ['"Jan"\n0.92', '"2023"\n0.79', '"the"\n0.64', '"20"\n0.58', '"-"\n0.31']
    valid = [False, True, False, True, False]

    for lane_name, y, constrained, outcome, outcome_color, outcome_face in lanes:
        ax.text(5, y + 4, lane_name, va="center", fontsize=9, fontweight="bold",
                color=BLUE if not constrained else ORANGE)
        for i, token in enumerate(candidate_tokens):
            x = 21 + i * 12.2
            is_valid = valid[i]
            disabled = constrained and not is_valid
            selected = (not constrained and i == 0) or (constrained and i == 1)
            face = BLUE_LIGHT if selected and not constrained else GREEN_LIGHT if selected else WHITE
            edge = BLUE if selected and not constrained else GREEN if selected else BORDER
            chip(
                ax,
                x,
                y,
                10.2,
                token,
                face=face,
                edge=edge,
                color=INK if not disabled else MUTED,
                alpha=0.38 if disabled else 1.0,
                strike=disabled,
            )
        arrow(ax, (82.5, y + 4), (87, y + 4), color=outcome_color, width=1.8)
        card(ax, 87.5, y - 0.2, 9.5, 8.4, face=outcome_face, edge=outcome_color, radius=2)
        ax.text(92.25, y + 4, outcome, ha="center", va="center", color=outcome_color,
                fontsize=8.8, fontweight="bold")

    ax.plot([5, 97], [51, 51], color=BORDER, lw=1)
    ax.text(50, 8, "Masking changes which tokens can win; it does not change their original scores.",
            ha="center", color=MUTED, fontsize=10)
    save(fig, "01_prompting_vs_constraints.png")


def regex_to_fsm() -> None:
    fig, ax = clean_canvas((15.2, 5.6))
    label(ax, 4, 92, "CHARACTER AUTOMATON")
    ax.text(4, 86, r"\d{4} - \d{2} - \d{2}", family="DejaVu Sans Mono", fontsize=13,
            color=INK, va="center")

    xs = np.linspace(7, 93, 11)
    edge_labels = ["digit", "digit", "digit", "digit", "-", "digit", "digit", "-", "digit", "digit"]
    group_ranges = [(0, 4, "YEAR"), (5, 7, "MONTH"), (8, 10, "DAY")]

    for start, end, name in group_ranges:
        left = xs[start] - 3.2
        right = xs[end] + 3.2
        ax.add_patch(
            FancyBboxPatch(
                (left, 34),
                right - left,
                35,
                boxstyle="round,pad=0.01,rounding_size=2.5",
                facecolor=PANEL,
                edgecolor="none",
            )
        )
        ax.text((left + right) / 2, 38.5, name, ha="center", va="center", fontsize=8.5,
                color=MUTED, fontweight="bold")

    for i, x in enumerate(xs):
        if i < 10:
            arrow(ax, (x + 1.45, 54), (xs[i + 1] - 1.45, 54), color=INK, width=1.25)
            ax.text((x + xs[i + 1]) / 2, 59.5, edge_labels[i], ha="center", fontsize=8.3,
                    color=ORANGE if edge_labels[i] == "-" else MUTED)
        final = i == 10
        ax.add_patch(
            Circle(
                (x, 54),
                1.55,
                facecolor=GREEN_LIGHT if final else WHITE,
                edgecolor=GREEN if final else INK,
                linewidth=1.45,
            )
        )
        if final:
            ax.add_patch(Circle((x, 54), 1.05, fill=False, edgecolor=GREEN, linewidth=1.0))
        ax.text(x, 47.5, f"s{i}", ha="center", fontsize=8.5, color=MUTED)

    # A tokenizer token can consume more than one character edge.
    arrow(ax, (xs[0], 67), (xs[2], 67), color=BLUE, width=2.0, curve=-0.24)
    card(ax, 10.2, 70, 11.2, 8.5, face=BLUE_LIGHT, edge=BLUE, radius=2)
    ax.text(15.8, 74.25, 'token "20"', ha="center", va="center", family="DejaVu Sans Mono",
            fontsize=9.5, color=BLUE, fontweight="bold")
    ax.text(50, 18, "The state replaces the generated prefix; token strings may traverse several edges.",
            ha="center", color=MUTED, fontsize=10)
    save(fig, "02_regex_to_fsm.png")


def precomputed_tables() -> None:
    fig, ax = clean_canvas((14.8, 6.3))
    label(ax, 4, 92, "DENSE LOOKUP AT STATE s4")

    states = ["s0", "s4", "s5", "s7", "s8", "s10"]
    tokens = ['"7"', '"20"', '"-"', '"01"', '"Jan"', "EOS"]
    allowed = np.array(
        [
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=int,
    )

    x0, y0 = 6, 23
    cell_w, cell_h = 6.2, 8.0
    for j, token in enumerate(tokens):
        ax.text(x0 + 10 + j * cell_w + cell_w / 2, y0 + 6 * cell_h + 5.2, token,
                ha="center", va="center", fontsize=8.6, family="DejaVu Sans Mono", color=MUTED)
    for i, state in enumerate(states):
        row_y = y0 + (5 - i) * cell_h
        highlighted = state == "s4"
        if highlighted:
            card(ax, x0 + 1, row_y - 0.5, 48.5, cell_h + 1, face=ORANGE_LIGHT,
                 edge=ORANGE, radius=1.5)
        ax.text(x0 + 6.5, row_y + cell_h / 2, state, ha="center", va="center", fontsize=9.2,
                color=ORANGE if highlighted else MUTED, fontweight="bold")
        for j in range(6):
            value = allowed[i, j]
            face = CYAN if value else "#E2E8F0"
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + 10 + j * cell_w + 1.2, row_y + 1.3),
                    3.8,
                    5.3,
                    boxstyle="round,pad=0.01,rounding_size=1",
                    facecolor=face,
                    edgecolor="none",
                )
            )
            ax.text(x0 + 10 + j * cell_w + 3.1, row_y + 3.95, "1" if value else "0",
                    ha="center", va="center", fontsize=8.7, color=WHITE if value else MUTED,
                    fontweight="bold")

    arrow(ax, (56.5, 55), (62, 55), color=ORANGE, width=2)
    label(ax, 63.5, 84, "GATHER + MASK + TRANSITION", color=ORANGE)
    card(ax, 63, 62, 31, 15, face=PANEL, edge=BORDER)
    ax.text(67, 71, "mask", fontsize=8.5, color=MUTED, fontweight="bold")
    for j, token in enumerate(tokens):
        x = 74 + j * 3.0
        enabled = j == 2
        ax.add_patch(Circle((x, 69.8), 0.95, facecolor=ORANGE if enabled else "#CBD5E1", edgecolor="none"))
    ax.text(79, 64.7, "only the separator survives", ha="center", fontsize=8.8, color=INK)

    card(ax, 63, 38, 31, 15, face=WHITE, edge=ORANGE)
    ax.text(67, 47, "argmax", fontsize=8.5, color=MUTED, fontweight="bold")
    chip(ax, 79, 41.4, 7, '"-"', face=ORANGE_LIGHT, edge=ORANGE, color=ORANGE)

    card(ax, 63, 14, 31, 15, face=WHITE, edge=GREEN)
    ax.text(67, 23, "transition", fontsize=8.5, color=MUTED, fontweight="bold")
    ax.text(78.5, 21.5, 'T[s4, "-"]  →  s5', ha="center", va="center",
            family="DejaVu Sans Mono", fontsize=10, color=GREEN, fontweight="bold")
    arrow(ax, (78.5, 61.5), (78.5, 53.5), color=ORANGE, width=1.6)
    arrow(ax, (78.5, 37.5), (78.5, 29.5), color=GREEN, width=1.6)
    save(fig, "03_precomputed_tables.png")


def accepting_not_terminal() -> None:
    fig, ax = clean_canvas((13.2, 4.7))
    label(ax, 4, 91, "REGEX a+ · ACCEPTING IS NOT TERMINAL", color=ORANGE)

    ax.text(5, 55, "start", fontsize=9, color=MUTED, va="center")
    arrow(ax, (12, 55), (24, 55), color=INK, width=1.5)

    ax.add_patch(Circle((29, 55), 3.2, facecolor=WHITE, edgecolor=INK, linewidth=1.5))
    ax.text(29, 55, "s0", ha="center", va="center", fontsize=10, color=INK)

    arrow(ax, (32.5, 55), (47, 55), color=BLUE, width=1.8)
    ax.text(
        39.5,
        61,
        '"a"',
        ha="center",
        fontsize=10,
        color=BLUE,
        family="DejaVu Sans Mono",
        fontweight="bold",
    )

    ax.add_patch(
        Circle((52, 55), 4.0, facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=1.8)
    )
    ax.add_patch(Circle((52, 55), 3.2, fill=False, edgecolor=GREEN, linewidth=1.2))
    ax.text(
        52,
        55,
        "s1",
        ha="center",
        va="center",
        fontsize=10,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        52, 42, "accepting", ha="center", fontsize=9, color=GREEN, fontweight="bold"
    )

    ax.add_patch(
        FancyArrowPatch(
            (48.5, 61.5),
            (55.5, 61.5),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.7,
            color=BLUE,
            connectionstyle="arc3,rad=-1.25",
        )
    )
    ax.text(
        52,
        80,
        'another "a"',
        ha="center",
        fontsize=9,
        color=BLUE,
        family="DejaVu Sans Mono",
        fontweight="bold",
    )

    arrow(ax, (56.5, 55), (72, 55), color=ORANGE, width=1.8)
    chip(ax, 73, 51, 13, "EOS", face=ORANGE_LIGHT, edge=ORANGE, color=ORANGE)
    ax.text(
        79.5, 42, "stop here", ha="center", fontsize=9, color=ORANGE, fontweight="bold"
    )

    card(ax, 87.5, 36, 9.5, 38, face=PANEL, edge=BORDER, radius=2)
    ax.text(92.25, 67, "LEGAL", ha="center", fontsize=8, color=MUTED, fontweight="bold")
    ax.text(
        92.25,
        57,
        '"a"',
        ha="center",
        fontsize=11,
        color=BLUE,
        family="DejaVu Sans Mono",
        fontweight="bold",
    )
    ax.text(
        92.25,
        47,
        "EOS",
        ha="center",
        fontsize=10,
        color=ORANGE,
        family="DejaVu Sans Mono",
        fontweight="bold",
    )

    ax.text(
        50,
        13,
        "The regex already matches at s1, but the language can still continue.",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    save(fig, "06_accepting_not_terminal.png")


def three_inner_loops() -> None:
    fig, ax = clean_canvas((15.2, 6.7))
    label(ax, 3, 94, "SAME QUESTION · WHICH TOKENS ARE ALLOWED?")

    columns = [
        (
            3,
            BLUE,
            BLUE_LIGHT,
            "ATTEMPT 1",
            "REGEX QUERY",
            "for token in vocab",
            "fullmatch(prefix + token,\npartial=True)",
            "24.9 ms",
            "prefix reparsed for every token",
        ),
        (
            36,
            CYAN,
            CYAN_LIGHT,
            "ATTEMPT 2",
            "FSM WALK",
            "for token in vocab",
            "walk_fsm(state, token)",
            "23.7 ms",
            "state retained; vocabulary loop remains",
        ),
        (
            69,
            GREEN,
            GREEN_LIGHT,
            "ATTEMPT 3",
            "ROW LOOKUP",
            "one tensor gather",
            "allowed[state]",
            "0.000542 ms",
            "constraint lookup only",
        ),
    ]

    for x, color, face, attempt, title, step_one, step_two, timing, takeaway in columns:
        card(ax, x, 21, 28, 66, face=WHITE, edge=color, radius=2.5, linewidth=1.4)
        label(ax, x + 3, 81, attempt, color=color)
        ax.text(x + 3, 73, title, fontsize=12, color=INK, fontweight="bold")

        card(ax, x + 3, 56, 22, 10, face=PANEL, edge=BORDER, radius=1.5)
        ax.text(
            x + 14,
            61,
            step_one,
            ha="center",
            va="center",
            fontsize=8.7,
            color=INK,
            family="DejaVu Sans Mono",
        )
        arrow(ax, (x + 14, 55.5), (x + 14, 49.5), color=color, width=1.5)
        card(ax, x + 3, 38, 22, 11, face=face, edge=color, radius=1.5)
        ax.text(
            x + 14,
            43.5,
            step_two,
            ha="center",
            va="center",
            fontsize=8.2,
            color=color,
            family="DejaVu Sans Mono",
            fontweight="bold",
        )
        ax.text(
            x + 14, 31, timing, ha="center", fontsize=13, color=color, fontweight="bold"
        )
        ax.text(x + 14, 25, takeaway, ha="center", fontsize=7.8, color=MUTED)

    arrow(ax, (31.5, 54), (35.5, 54), color=MUTED, width=1.3)
    arrow(ax, (64.5, 54), (68.5, 54), color=MUTED, width=1.3)
    ax.text(
        50,
        10,
        "The row lookup excludes dense masking, argmax, and model execution.",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    save(fig, "07_three_inner_loops.png")


def benchmark_results() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8), gridspec_kw={"wspace": 0.44})
    fig.patch.set_facecolor(PAPER)

    # Constraint lookup uses a log scale because the precomputed row is orders smaller.
    ax = axes[0]
    methods = ["partial regex", "FSM rescan", "FSM state", "table row"]
    values = [24.908, 50.114, 23.697, 0.000542]
    colors = ["#94A3B8", ORANGE, CYAN, BLUE]
    ax.barh(methods[::-1], values[::-1], color=colors[::-1], height=0.52)
    ax.set_xscale("log")
    ax.set_xlim(0.0002, 100)
    ax.set_title("Constraint lookup", loc="left", fontsize=11, fontweight="bold", pad=14)
    ax.set_xlabel("milliseconds · log scale", fontsize=8.5)
    ax.grid(axis="x", which="major", color=BORDER, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, value in enumerate(values[::-1]):
        text = f"{value:.6f}" if value < 0.01 else f"{value:.1f}"
        ax.text(value * 1.22, i, text, va="center", fontsize=8.2, color=INK)

    def dumbbell(
        axis: plt.Axes,
        title: str,
        before: float,
        after: float,
        before_label: str,
        after_label: str,
        speedup: str,
    ) -> None:
        axis.set_xlim(0, before * 1.18)
        axis.set_ylim(-0.8, 1.1)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=14)
        axis.plot([after, before], [0, 0], color=BORDER, lw=5, solid_capstyle="round")
        axis.scatter([before], [0], s=150, color=ORANGE, zorder=3)
        axis.scatter([after], [0], s=150, color=GREEN, zorder=3)
        axis.text(before, -0.24, before_label, ha="center", va="top", fontsize=8.5, color=ORANGE,
                  fontweight="bold")
        axis.text(after, -0.24, after_label, ha="center", va="top", fontsize=8.5, color=GREEN,
                  fontweight="bold")
        axis.text((before + after) / 2, 0.34, speedup, ha="center", va="center", fontsize=15,
                  color=GREEN, fontweight="bold")
        axis.text((before + after) / 2, 0.18, "faster", ha="center", va="center", fontsize=8,
                  color=MUTED)
        axis.set_yticks([])
        axis.set_xlabel("median milliseconds", fontsize=8.5)
        axis.grid(axis="x", color=BORDER, linewidth=0.8)
        axis.set_axisbelow(True)

    dumbbell(axes[1], "KV cache", 3592.51, 456.41, "3.59 s", "0.46 s", "7.87×")
    dumbbell(axes[2], "Batching", 2951.45, 1684.41, "2.95 s", "1.68 s", "1.75×")

    for ax in axes:
        ax.set_facecolor(PAPER)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=8.5)

    fig.text(
        0.5,
        -0.01,
        "SmolLM2-135M-Instruct · CPU · four PyTorch threads · medians exclude model loading",
        ha="center",
        color=MUTED,
        fontsize=8.8,
    )
    save(fig, "04_benchmark_results.png")


def batched_state_tracking() -> None:
    fig, ax = clean_canvas((14.6, 6.0))
    label(ax, 4, 92, "ONE REGEX TABLE · THREE INDEPENDENT ROWS")

    card(ax, 37, 78, 26, 12, face=BLUE_LIGHT, edge=BLUE)
    ax.text(50, 84, "shared allowed[state, token]", ha="center", va="center",
            family="DejaVu Sans Mono", fontsize=10, color=BLUE, fontweight="bold")

    rows = [
        ("row 0", "2023", "s4", '"-"', "s5", ORANGE),
        ("row 1", "19", "s2", '"84"', "s4", CYAN),
        ("row 2", "2024-0", "s6", '"7"', "s7", GREEN),
    ]
    ys = [59, 38, 17]
    for (row, prefix, state, token, next_state, color), y in zip(rows, ys):
        ax.text(4, y + 5, row, fontsize=8.5, color=MUTED, fontweight="bold", va="center")
        card(ax, 11, y, 19, 10, face=WHITE, edge=BORDER)
        ax.text(20.5, y + 5, prefix, ha="center", va="center", family="DejaVu Sans Mono",
                fontsize=10, color=INK)
        arrow(ax, (30.5, y + 5), (36, y + 5), color=MUTED)
        card(ax, 36.5, y, 9, 10, face=PANEL, edge=color)
        ax.text(41, y + 5, state, ha="center", va="center", fontsize=10, color=color,
                fontweight="bold")
        arrow(ax, (45.8, y + 5), (54, y + 5), color=color)
        card(ax, 54.5, y, 14, 10, face=PANEL, edge=BORDER)
        ax.text(61.5, y + 5, f"mask[{state}]", ha="center", va="center",
                family="DejaVu Sans Mono", fontsize=9, color=INK)
        arrow(ax, (68.8, y + 5), (76, y + 5), color=MUTED)
        chip(ax, 76.5, y + 1, 8, token, face=WHITE, edge=color, color=color)
        arrow(ax, (84.8, y + 5), (90, y + 5), color=color)
        card(ax, 90.5, y, 6.5, 10, face=WHITE, edge=color)
        ax.text(93.75, y + 5, next_state, ha="center", va="center", fontsize=9.5,
                color=color, fontweight="bold")
        arrow(ax, (50, 78), (61.5, y + 10.5), color="#94A3B8", width=1.0)

    ax.text(50, 5, "Each row carries its own state, finished flag, and generated length.",
            ha="center", color=MUTED, fontsize=9.8)
    save(fig, "05_batched_state_tracking.png")


if __name__ == "__main__":
    configure()
    cover()
    prompting_vs_constraints()
    regex_to_fsm()
    accepting_not_terminal()
    precomputed_tables()
    three_inner_loops()
    benchmark_results()
    batched_state_tracking()
    print(f"Wrote diagrams to {OUT}")
