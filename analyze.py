"""
Compute and display accuracy tables from completed experiment results.

Usage:
  uv run python analyze.py
  uv run python analyze.py --csv results.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.jsonl"
JUDGMENTS_FILE = DATA_DIR / "judgments.jsonl"

console = Console()


def load_judgments():
    from models import Judgment

    if not JUDGMENTS_FILE.exists():
        console.print("[red]No judgments file found.[/red]")
        sys.exit(1)
    results = []
    for line in JUDGMENTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            results.append(Judgment.model_validate_json(line))
    return results


def short_model_name(model: str) -> str:
    # "meta-llama/Llama-3.3-70B-Instruct-Turbo" -> "70B"
    for size in ("1B", "3B", "8B", "70B"):
        if size in model:
            return size
    return model.split("/")[-1]


def main(csv_out: str | None = None) -> None:
    judgments = load_judgments()
    console.print(f"\n[bold]Loaded {len(judgments)} judgments.[/bold]\n")

    # Group by (judge_model, query_budget)
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for j in judgments:
        groups[(j.judge_model, j.query_budget)].append(j)

    judge_models = sorted({j.judge_model for j in judgments}, key=lambda m: short_model_name(m))
    budgets = sorted({j.query_budget for j in judgments})

    # --- Primary table: accuracy by judge × budget ---
    acc_table = Table(title="Judge Accuracy (% correct verdicts)")
    acc_table.add_column("Judge", style="bold")
    for b in budgets:
        acc_table.add_column(f"Budget {b}", justify="right")

    rows: list[dict] = []
    for model in judge_models:
        cells = []
        for b in budgets:
            group = groups[(model, b)]
            if not group:
                cells.append("-")
                rows.append({"judge": short_model_name(model), "budget": b, "accuracy": None, "n": 0})
                continue
            acc = sum(j.verdict_correct for j in group) / len(group)
            cells.append(f"{acc:.1%} (n={len(group)})")
            rows.append({"judge": short_model_name(model), "budget": b, "accuracy": acc, "n": len(group)})
        acc_table.add_row(short_model_name(model), *cells)

    console.print(acc_table)

    # --- Secondary table: dishonest-debater win rate ---
    dis_table = Table(title="Dishonest Debater Win Rate (% incorrect verdicts)")
    dis_table.add_column("Judge", style="bold")
    for b in budgets:
        dis_table.add_column(f"Budget {b}", justify="right")

    for model in judge_models:
        cells = []
        for b in budgets:
            group = groups[(model, b)]
            if not group:
                cells.append("-")
                continue
            dishonest_wins = sum(1 for j in group if not j.verdict_correct)
            rate = dishonest_wins / len(group)
            cells.append(f"{rate:.1%}")
        dis_table.add_row(short_model_name(model), *cells)

    console.print(dis_table)

    # --- Tertiary: marginal accuracy gain per budget step ---
    eff_table = Table(title="Marginal Accuracy Gain (vs. budget 0)")
    eff_table.add_column("Judge", style="bold")
    for b in budgets[1:]:
        eff_table.add_column(f"Budget {b}", justify="right")

    for model in judge_models:
        base_group = groups[(model, 0)]
        if not base_group:
            continue
        base_acc = sum(j.verdict_correct for j in base_group) / len(base_group)
        cells = []
        for b in budgets[1:]:
            group = groups[(model, b)]
            if not group:
                cells.append("-")
                continue
            acc = sum(j.verdict_correct for j in group) / len(group)
            delta = acc - base_acc
            sign = "+" if delta >= 0 else ""
            cells.append(f"{sign}{delta:.1%}")
        eff_table.add_row(short_model_name(model), *cells)

    console.print(eff_table)

    # --- CSV export ---
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["judge", "budget", "accuracy", "n"])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"\n[green]Saved to {csv_out}[/green]")


if __name__ == "__main__":
    csv_path = None
    args = sys.argv[1:]
    if "--csv" in args:
        idx = args.index("--csv")
        if idx + 1 < len(args):
            csv_path = args[idx + 1]
    main(csv_out=csv_path)
