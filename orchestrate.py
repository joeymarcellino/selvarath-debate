"""
CLI entry point for the debate experiment.

Commands:
  debate preview           — show counts and cost estimates, no API calls
  debate debates           — generate all debate transcripts
  debate judges            — run all judge evaluations over existing transcripts
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import debate as debate_module
import judge as judge_module
from models import Question, Transcript, Judgment

app = typer.Typer(help="Debate experiment pipeline")
console = Console()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.jsonl"
JUDGMENTS_FILE = DATA_DIR / "judgments.jsonl"
PROTOCOL_FILE = ROOT / "experiment_protocol.json"
WORLD_SPECS_DIR = ROOT / "world_specs"
QUESTIONS_DIR = ROOT / "questions"

WORLD_FILES = {
    "carath_norn": WORLD_SPECS_DIR / "carath_norn.txt",
    "selvarath": WORLD_SPECS_DIR / "selvarath.txt",
    "vethun_sarak": WORLD_SPECS_DIR / "vethun_sarak.txt",
}

QUESTION_FILES = {
    "carath_norn": QUESTIONS_DIR / "carath_norn_questions.json",
    "selvarath": QUESTIONS_DIR / "selvarath_questions.json",
    "vethun_sarak": QUESTIONS_DIR / "vethun_sarak_questions.json",
}


def load_protocol() -> dict:
    return json.loads(PROTOCOL_FILE.read_text())


def load_all_questions() -> list[Question]:
    questions = []
    for path in QUESTION_FILES.values():
        raw = json.loads(path.read_text())
        questions.extend(Question.model_validate(q) for q in raw)
    return questions


def load_world(world: str) -> str:
    return WORLD_FILES[world].read_text()


def load_transcripts() -> list[Transcript]:
    if not TRANSCRIPTS_FILE.exists():
        return []
    results = []
    for line in TRANSCRIPTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            results.append(Transcript.model_validate_json(line))
    return results


def load_judgments() -> list[Judgment]:
    if not JUDGMENTS_FILE.exists():
        return []
    results = []
    for line in JUDGMENTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            results.append(Judgment.model_validate_json(line))
    return results


def append_transcript(t: Transcript) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with TRANSCRIPTS_FILE.open("a") as f:
        f.write(t.model_dump_json() + "\n")


def append_judgment(j: Judgment) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with JUDGMENTS_FILE.open("a") as f:
        f.write(j.model_dump_json() + "\n")


@app.command()
def preview() -> None:
    """Show question counts, work remaining, and cost estimates."""
    protocol = load_protocol()
    questions = load_all_questions()
    existing_transcripts = load_transcripts()
    existing_judgments = load_judgments()

    n_questions = len(questions)
    transcripts_per_q = protocol["protocol"]["debate_phase"]["transcripts_per_question"]
    total_transcripts = n_questions * transcripts_per_q
    done_transcripts = len(existing_transcripts)

    judge_models: list[str] = protocol["protocol"]["models"]["judges"]
    budgets: list[int] = protocol["protocol"]["judge_phase"]["query_budgets"]
    total_judgment_cells = total_transcripts * len(judge_models) * len(budgets)
    done_judgments = len(existing_judgments)

    # Question counts per world
    world_counts: dict[str, int] = {}
    for q in questions:
        world_counts[q.world] = world_counts.get(q.world, 0) + 1

    console.print("\n[bold]Debate Experiment — Preview[/bold]\n")

    t = Table(title="Questions by world")
    t.add_column("World")
    t.add_column("Questions", justify="right")
    t.add_column("Transcripts (×3)", justify="right")
    for world, count in sorted(world_counts.items()):
        t.add_row(world, str(count), str(count * transcripts_per_q))
    t.add_row("[bold]Total[/bold]", str(n_questions), str(total_transcripts), style="bold")
    console.print(t)

    console.print(f"\n[bold]Transcripts:[/bold] {done_transcripts}/{total_transcripts} done "
                  f"({total_transcripts - done_transcripts} remaining)")
    console.print(f"[bold]Judgment cells:[/bold] {done_judgments}/{total_judgment_cells} done "
                  f"({total_judgment_cells - done_judgments} remaining)")

    console.print("\n[bold]Cost estimates (from protocol):[/bold]")
    cost = protocol["protocol"]["cost_estimate"]
    for key, val in cost.items():
        console.print(f"  {key}: {val}")

    console.print("\n[bold]Judge models:[/bold] " + ", ".join(judge_models))
    console.print("[bold]Query budgets:[/bold] " + ", ".join(str(b) for b in budgets))
    console.print()


@app.command()
def debates(
    dry_run: bool = typer.Option(False, "--dry-run", help="No API calls; use DRY_RUN env var"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only first N questions"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent API calls"),
) -> None:
    """Generate all debate transcripts."""
    if dry_run:
        os.environ["DRY_RUN"] = "1"

    asyncio.run(_run_debates(limit=limit, concurrency=concurrency, dry_run=dry_run))


async def _run_debates(limit: int | None, concurrency: int, dry_run: bool) -> None:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    protocol = load_protocol()
    questions = load_all_questions()
    if limit is not None:
        questions = questions[:limit]

    transcripts_per_q: int = protocol["protocol"]["debate_phase"]["transcripts_per_question"]

    # Build set of completed keys
    existing = {(t.question_id, t.transcript_index) for t in load_transcripts()}

    work = [
        (q, idx)
        for q in questions
        for idx in range(transcripts_per_q)
        if (q.id, idx) not in existing
    ]

    if not work:
        console.print("[green]All transcripts already generated.[/green]")
        return

    console.print(f"[bold]Generating {len(work)} transcripts[/bold] "
                  f"({'dry run' if dry_run else 'live'}), concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    errors = 0

    async def run_one(q: Question, idx: int) -> None:
        nonlocal completed, errors
        async with sem:
            try:
                world_doc = load_world(q.world)
                t = await debate_module.generate_transcript(q, world_doc, idx, protocol)
                if not dry_run:
                    append_transcript(t)
                completed += 1
            except Exception as e:
                errors += 1
                console.print(f"[red]Error on {q.id}/{idx}: {e}[/red]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcripts", total=len(work))

        async def tracked(q: Question, idx: int) -> None:
            await run_one(q, idx)
            progress.advance(task)

        await asyncio.gather(*[tracked(q, idx) for q, idx in work])

    console.print(f"\n[green]Done.[/green] Completed: {completed}, Errors: {errors}")


@app.command()
def judges(
    dry_run: bool = typer.Option(False, "--dry-run", help="No API calls"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only first N transcripts"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Run only this judge model"),
    budget: Optional[int] = typer.Option(None, "--budget", help="Run only this query budget"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent API calls"),
) -> None:
    """Run all judge evaluations over existing transcripts."""
    if dry_run:
        os.environ["DRY_RUN"] = "1"

    asyncio.run(_run_judges(
        limit=limit,
        concurrency=concurrency,
        dry_run=dry_run,
        filter_judge=judge_model,
        filter_budget=budget,
    ))


async def _run_judges(
    limit: int | None,
    concurrency: int,
    dry_run: bool,
    filter_judge: str | None,
    filter_budget: int | None,
) -> None:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    protocol = load_protocol()
    transcripts = load_transcripts()
    if limit is not None:
        transcripts = transcripts[:limit]

    judge_models: list[str] = protocol["protocol"]["models"]["judges"]
    budgets: list[int] = protocol["protocol"]["judge_phase"]["query_budgets"]

    if filter_judge is not None:
        judge_models = [m for m in judge_models if m == filter_judge]
        if not judge_models:
            console.print(f"[red]Unknown judge model: {filter_judge}[/red]")
            raise typer.Exit(1)

    if filter_budget is not None:
        budgets = [b for b in budgets if b == filter_budget]
        if not budgets:
            console.print(f"[red]Budget {filter_budget} not in protocol budgets.[/red]")
            raise typer.Exit(1)

    existing = {
        (j.question_id, j.transcript_index, j.judge_model, j.query_budget)
        for j in load_judgments()
    }

    work = [
        (t, jm, b)
        for t in transcripts
        for jm in judge_models
        for b in budgets
        if (t.question_id, t.transcript_index, jm, b) not in existing
    ]

    if not work:
        console.print("[green]All judgments already completed.[/green]")
        return

    console.print(f"[bold]Running {len(work)} judgments[/bold] "
                  f"({'dry run' if dry_run else 'live'}), concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    errors = 0

    world_docs: dict[str, str] = {}
    for t in transcripts:
        if t.world not in world_docs:
            world_docs[t.world] = load_world(t.world)

    async def run_one(t: Transcript, jm: str, b: int) -> None:
        nonlocal completed, errors
        async with sem:
            try:
                j = await judge_module.run_judgment(t, world_docs[t.world], jm, b, protocol)
                if not dry_run:
                    append_judgment(j)
                completed += 1
            except Exception as e:
                errors += 1
                console.print(f"[red]Error on {t.question_id}/{t.transcript_index}/{jm}/budget={b}: {e}[/red]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Judgments", total=len(work))

        async def tracked(t: Transcript, jm: str, b: int) -> None:
            await run_one(t, jm, b)
            progress.advance(task)

        await asyncio.gather(*[tracked(t, jm, b) for t, jm, b in work])

    console.print(f"\n[green]Done.[/green] Completed: {completed}, Errors: {errors}")


if __name__ == "__main__":
    app()
