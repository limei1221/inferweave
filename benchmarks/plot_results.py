"""Render the README figures from the published benchmark table.

Run: uv run --no-project --with matplotlib python benchmarks/plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/benchmark-results-2026-10-03.md"
DESTINATION = ROOT / "assets/benchmarks"


def read_results():
    results = {}
    model = None
    for line in SOURCE.read_text().splitlines():
        if line.startswith("## "):
            model = line.removeprefix("## ").split(" (")[0]
            results[model] = {"lean-vllm": [], "vllm": []}
        if not line.startswith(("| lean-vllm |", "| vllm |")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        system, arm = cells[:2]
        if arm == "async-scheduling=False":
            continue
        # Offered rate, goodput (including drain), median TPOT converted to milliseconds.
        results[model][system].append((float(cells[2]), float(cells[6]), 1000 * float(cells[9])))
    return results


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "savefig.facecolor": "white",
        }
    )
    DESTINATION.mkdir(parents=True, exist_ok=True)
    results = read_results()
    for model, filename in (
        ("Qwen3-8B", "qwen3-8b.png"),
        ("DeepSeek-V2-Lite-Chat", "deepseek-v2-lite.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.29, top=0.70, wspace=0.30)
        fig.suptitle(model, x=0.075, y=0.97, ha="left", fontsize=19, fontweight="bold")
        fig.text(
            0.075, 0.865, "H100 80 GB  |  3 Oct 2026  |  1,000 requests / point  |  Async scheduling on", fontsize=10
        )
        for system, label, color, marker, style in (
            ("lean-vllm", "lean-vLLM", "#2563eb", "o", "-"),
            ("vllm", "vLLM 0.26.0", "#c05621", "s", "--"),
        ):
            rows = sorted(results[model][system])
            assert [row[0] for row in rows] == [1, 24, 32, 48, 64], (model, system)
            rates, goodput, tpot = zip(*rows)
            for ax, values in zip(axes, (goodput, tpot)):
                ax.plot(
                    rates, values, label=label, color=color, marker=marker, linestyle=style, linewidth=2.2, markersize=6
                )
        for ax, title, ylabel, ymax in zip(
            axes,
            ("Throughput · higher is better", "Token latency · lower is better"),
            ("Completed requests / s", "Median time per output token (ms)"),
            (35, 60),
        ):
            ax.set_title(title, loc="left", fontsize=12, pad=12)
            ax.set_xlabel("Offered load (requests / s)", labelpad=8)
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 24, 32, 48, 64])
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
            ax.set_axisbelow(True)
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=2, frameon=False
        )
        fig.text(
            0.075,
            0.025,
            "Throughput includes queue drain; no latency cutoff. One run per point. Source: docs/benchmark-results-2026-10-03.md",
            fontsize=8,
            color="#64748b",
        )
        output = DESTINATION / filename
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
