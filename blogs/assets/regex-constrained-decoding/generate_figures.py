"""Generate the technical figures for the regex-constrained decoding post."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent

NAVY = "#102A43"
BLUE = "#2F6690"
TEAL = "#4F9D9D"
PALE_TEAL = "#D9ECE8"
CORAL = "#EF8354"
PALE_CORAL = "#FBE1D5"
CREAM = "#FBF7EF"
SLATE = "#627D98"
LIGHT = "#E8EEF2"
WHITE = "#FFFFFF"
RED = "#C94C4C"
GREEN = "#2F855A"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, *, facecolor=WHITE, edgecolor=LIGHT,
                fontsize=11, color=NAVY, linewidth=1.5, radius=0.08):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color)
    return patch


def arrow(ax, start, end, *, color=SLATE, linewidth=1.7, mutation_scale=13,
          connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(patch)
    return patch


def prompting_vs_masking() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"wspace": 0.28})
    tokens = ["Jan", "2023", "the", "20", "-", "7"]
    scores = np.array([8.8, 7.5, 6.9, 6.3, 4.6, 3.8])
    colors = [CORAL, TEAL, CORAL, TEAL, CORAL, TEAL]

    for ax in axes:
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.grid(axis="x", color=LIGHT, linewidth=0.8, zorder=0)

    axes[0].barh(tokens[::-1], scores[::-1], color=colors[::-1], height=0.62, zorder=3)
    axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("model score (illustrative)")
    axes[0].set_title("Prompting changes preferences")
    axes[0].text(
        0,
        -0.20,
        "The model can still choose an invalid continuation.",
        transform=axes[0].transAxes,
        color=SLATE,
        fontsize=10.5,
        va="top",
    )

    masked_scores = scores.copy()
    valid = np.array([False, True, False, True, False, True])
    masked_scores[~valid] = 0
    masked_colors = np.where(valid, TEAL, LIGHT)
    axes[1].barh(tokens[::-1], masked_scores[::-1], color=masked_colors[::-1], height=0.62, zorder=3)
    for index, (token, is_valid) in enumerate(zip(tokens[::-1], valid[::-1])):
        if not is_valid:
            axes[1].text(0.18, index, r"$-\infty$", va="center", ha="left", color=RED, fontsize=11)
    axes[1].set_xlim(0, 10)
    axes[1].set_xlabel("score after the regex mask")
    axes[1].set_title("Constrained decoding removes choices")
    axes[1].text(
        0,
        -0.20,
        "Invalid tokens have zero probability after softmax.",
        transform=axes[1].transAxes,
        color=SLATE,
        fontsize=10.5,
        va="top",
    )

    fig.suptitle("A prompt asks for a format; a decoding mask enforces it", fontsize=17,
                 fontweight="bold", y=1.04, color=NAVY)
    save(fig, "01_prompting_vs_constraints.png")


def regex_to_fsm() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 4.7))
    ax.set_xlim(-0.8, 10.9)
    ax.set_ylim(-1.65, 1.55)
    ax.axis("off")

    labels = ["start", "1 digit", "2 digits", "3 digits", "year", "-", "1 digit",
              "month", "-", "1 digit", "date"]
    xs = np.arange(11)
    for i in range(10):
        arrow(ax, (xs[i] + 0.24, 0), (xs[i + 1] - 0.24, 0), color=BLUE, linewidth=1.6)
        edge_label = "digit" if i not in (4, 7) else "-"
        ax.text((xs[i] + xs[i + 1]) / 2, 0.35, edge_label, ha="center", va="center",
                fontsize=9.5, color=SLATE)

    for i, (x, label) in enumerate(zip(xs, labels)):
        fill = PALE_TEAL if i == 10 else WHITE
        edge = TEAL if i == 10 else NAVY
        ax.add_patch(Circle((x, 0), 0.25, facecolor=fill, edgecolor=edge, linewidth=2))
        if i == 10:
            ax.add_patch(Circle((x, 0), 0.18, facecolor="none", edgecolor=edge, linewidth=1.3))
        ax.text(x, -0.52, f"s{i}", ha="center", va="center", fontweight="bold", fontsize=9.5)
        ax.text(x, -0.86, label, ha="center", va="center", fontsize=8.4, color=SLATE)

    ax.text(-0.72, 0.78, r"regex:  $\backslash d\{4\}$-$\backslash d\{2\}$-$\backslash d\{2\}$",
            fontsize=14, fontweight="bold", color=NAVY)

    # Show that one tokenizer token can traverse more than one character edge.
    arrow(ax, (0.05, 0.34), (1.95, 0.34), color=CORAL, linewidth=2.2,
          connectionstyle="arc3,rad=-0.35")
    rounded_box(ax, (0.67, 0.78), 0.72, 0.38, 'token "20"', facecolor=PALE_CORAL,
                edgecolor=CORAL, fontsize=9.5, color=NAVY, radius=0.09)
    ax.text(1.03, 1.33, "one token, two transitions", ha="center", fontsize=9.3,
            color=CORAL, fontweight="bold")

    ax.text(5.0, -1.43,
            "The state is the only prefix information needed to decide which token strings remain valid.",
            ha="center", fontsize=11.2, color=SLATE)
    fig.suptitle("The regex becomes a deterministic state machine", fontsize=17,
                 fontweight="bold", y=0.99, color=NAVY)
    save(fig, "02_regex_to_fsm.png")


def precomputed_tables() -> None:
    states = ["s0\nstart", "s4\nyear", "s5\nafter -", "s7\nmonth", "s8\nafter -", "s10\nfinal"]
    tokens = ['"7"', '"20"', '"-"', '"01"', '"Jan"', "EOS"]
    next_state = np.array(
        [
            [1, 2, -1, 2, -1, -1],
            [-1, -1, 5, -1, -1, -1],
            [6, 7, -1, 7, -1, -1],
            [-1, -1, 8, -1, -1, -1],
            [9, 10, -1, 10, -1, -1],
            [-1, -1, -1, -1, -1, 10],
        ],
        dtype=int,
    )
    allowed = next_state >= 0

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.imshow(allowed, cmap=plt.matplotlib.colors.ListedColormap([LIGHT, TEAL]), aspect="auto")
    ax.set_xticks(range(len(tokens)), tokens)
    ax.set_yticks(range(len(states)), states)
    ax.set_title("allowed[state, token]")
    for i in range(allowed.shape[0]):
        for j in range(allowed.shape[1]):
            ax.text(j, i, "✓" if allowed[i, j] else "×", ha="center", va="center",
                    color=WHITE if allowed[i, j] else SLATE, fontweight="bold", fontsize=13)
    ax.tick_params(length=0)
    ax.set_xlabel("token string")
    ax.set_ylabel("current FSM state")

    ax = axes[1]
    ax.imshow(allowed, cmap=plt.matplotlib.colors.ListedColormap([LIGHT, PALE_CORAL]), aspect="auto")
    ax.set_xticks(range(len(tokens)), tokens)
    ax.set_yticks(range(len(states)), states)
    ax.set_title("transition[state, token]")
    for i in range(next_state.shape[0]):
        for j in range(next_state.shape[1]):
            value = "—" if next_state[i, j] < 0 else f"s{next_state[i, j]}"
            ax.text(j, i, value, ha="center", va="center", color=NAVY if value != "—" else SLATE,
                    fontweight="bold" if value != "—" else "normal", fontsize=11)
    ax.tick_params(length=0)
    ax.set_xlabel("chosen token string")
    ax.set_ylabel("current FSM state")

    fig.suptitle("Precomputation turns Python work into two tensor lookups", fontsize=17,
                 fontweight="bold", y=1.02, color=NAVY)
    fig.text(0.5, -0.01,
             "The mask answers what may be sampled; the transition table advances every batch row independently.",
             ha="center", color=SLATE, fontsize=10.8)
    save(fig, "03_precomputed_tables.png")


def benchmark_results() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), gridspec_kw={"wspace": 0.48})

    methods = ["Partial regex", "FSM: rescan prefix", "FSM: track state", "Precomputed row"]
    values = [24.908, 50.114, 23.697, 0.000542]
    colors = [BLUE, CORAL, TEAL, NAVY]
    ax = axes[0]
    bars = ax.barh(methods[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(0.0002, 100)
    ax.set_xlabel("median milliseconds per mask step (log scale)")
    ax.set_title("Precompute the constraint")
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", color=LIGHT, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values[::-1]):
        label = f"{value:.6f} ms" if value < 0.01 else f"{value:.1f} ms"
        ax.text(value * 1.22, bar.get_y() + bar.get_height() / 2, label, va="center",
                fontsize=9.2, color=NAVY)

    ax = axes[1]
    labels = ["No cache", "KV cache"]
    values_cache = [3592.51, 456.41]
    bars = ax.bar(labels, values_cache, color=[CORAL, TEAL], width=0.58)
    ax.set_ylabel("median milliseconds")
    ax.set_title("Cache model attention")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    for bar, value in zip(bars, values_cache):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 95, f"{value / 1000:.2f} s",
                ha="center", va="bottom", fontweight="bold", color=NAVY)
    ax.text(0.5, 0.84, "7.87×", transform=ax.transAxes, ha="center", va="center",
            fontsize=18, fontweight="bold", color=GREEN)

    ax = axes[2]
    labels = ["4 sequential", "batch of 4"]
    values_batch = [2951.45, 1684.41]
    bars = ax.bar(labels, values_batch, color=[CORAL, TEAL], width=0.58)
    ax.set_ylabel("median milliseconds")
    ax.set_title("Batch independent rows")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    for bar, value in zip(bars, values_batch):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 80, f"{value / 1000:.2f} s",
                ha="center", va="bottom", fontweight="bold", color=NAVY)
    ax.text(0.5, 0.84, "1.75×", transform=ax.transAxes, ha="center", va="center",
            fontsize=18, fontweight="bold", color=GREEN)

    fig.suptitle("Three optimizations act on three different costs", fontsize=17,
                 fontweight="bold", y=1.03, color=NAVY)
    fig.text(0.5, -0.025,
             "SmolLM2-135M-Instruct, 49,152-token vocabulary, CPU, four PyTorch threads; medians exclude model loading.",
             ha="center", color=SLATE, fontsize=10.4)
    save(fig, "04_benchmark_results.png")


def batched_state_tracking() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.7))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    rounded_box(ax, (0.35, 4.85), 2.0, 0.72, "model logits\n[B, V]", facecolor=WHITE,
                edgecolor=BLUE, fontsize=11.5)
    rounded_box(ax, (3.05, 4.85), 2.1, 0.72, "FSM states\n[B]", facecolor=WHITE,
                edgecolor=TEAL, fontsize=11.5)
    rounded_box(ax, (5.85, 4.85), 2.3, 0.72, "allowed[states]\n[B, V]", facecolor=PALE_TEAL,
                edgecolor=TEAL, fontsize=11.5)
    rounded_box(ax, (8.85, 4.85), 2.0, 0.72, "masked argmax\n[B]", facecolor=PALE_CORAL,
                edgecolor=CORAL, fontsize=11.5)
    rounded_box(ax, (11.55, 4.85), 1.9, 0.72, "next states\n[B]", facecolor=WHITE,
                edgecolor=NAVY, fontsize=11.5)
    for left, right in [((2.35, 5.21), (3.05, 5.21)), ((5.15, 5.21), (5.85, 5.21)),
                        ((8.15, 5.21), (8.85, 5.21)), ((10.85, 5.21), (11.55, 5.21))]:
        arrow(ax, left, right, color=SLATE)

    rows = [
        ("row 0", "2023", "s4", '"-"', "s5"),
        ("row 1", "19", "s2", '"84"', "s4"),
        ("row 2", "2024-0", "s6", '"7"', "s7"),
    ]
    ys = [3.55, 2.35, 1.15]
    for (row, prefix, state, token, nxt), y in zip(rows, ys):
        ax.text(0.35, y, row, va="center", fontweight="bold", color=SLATE)
        rounded_box(ax, (1.2, y - 0.33), 2.55, 0.66, prefix, facecolor=WHITE,
                    edgecolor=LIGHT, fontsize=11)
        rounded_box(ax, (4.15, y - 0.33), 1.15, 0.66, state, facecolor=PALE_TEAL,
                    edgecolor=TEAL, fontsize=11, color=NAVY)
        rounded_box(ax, (7.05, y - 0.33), 1.25, 0.66, token, facecolor=PALE_CORAL,
                    edgecolor=CORAL, fontsize=11, color=NAVY)
        rounded_box(ax, (10.15, y - 0.33), 1.15, 0.66, nxt, facecolor=WHITE,
                    edgecolor=NAVY, fontsize=11)
        arrow(ax, (3.75, y), (4.15, y), color=SLATE)
        arrow(ax, (5.3, y), (7.05, y), color=SLATE)
        arrow(ax, (8.3, y), (10.15, y), color=SLATE)

    ax.text(6.9, 0.28,
            "A batch shares the regex tables, but each row carries its own state and finished flag.",
            ha="center", color=SLATE, fontsize=11)
    fig.suptitle("Batching is vectorized state bookkeeping", fontsize=17,
                 fontweight="bold", y=0.99, color=NAVY)
    save(fig, "05_batched_state_tracking.png")


if __name__ == "__main__":
    setup()
    prompting_vs_masking()
    regex_to_fsm()
    precomputed_tables()
    benchmark_results()
    batched_state_tracking()
    print(f"Wrote figures to {OUT}")
