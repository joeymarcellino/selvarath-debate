import hashlib
import logging
import random

import api
from models import Judgment, OracleExchange, Transcript

logger = logging.getLogger(__name__)


def make_seed(*parts: str | int) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _format_transcript(transcript: Transcript, position_a_is_correct: bool) -> tuple[str, str, str]:
    if position_a_is_correct:
        position_a = transcript.correct_answer
        position_b = transcript.wrong_answer
    else:
        position_a = transcript.wrong_answer
        position_b = transcript.correct_answer

    turns = []
    for i, turn in enumerate(transcript.debate_transcript):
        # Assign debate labels: A = correct-side debater, B = wrong-side debater
        # (from the judge's perspective, positions are just A and B)
        if position_a_is_correct:
            label = "Debater A" if turn.speaker == "honest" else "Debater B"
        else:
            label = "Debater B" if turn.speaker == "honest" else "Debater A"
        turns.append(f"Turn {i + 1} ({label}):\n{turn.text}")

    debate_text = "\n\n".join(turns)
    return position_a, position_b, debate_text


def _format_previous_queries(exchanges: list[OracleExchange]) -> str:
    if not exchanges:
        return "No queries submitted yet."
    lines = []
    for i, ex in enumerate(exchanges, 1):
        lines.append(f"Query {i}: {ex.query}\nResult: {ex.response}")
    return "\n\n".join(lines)


def _parse_verdict(text: str) -> tuple[str, int, str]:
    """
    Returns (verdict, confidence, reasoning).
    verdict is "Position A" or "Position B".
    Falls back to scanning the full text if structured parse fails.
    """
    verdict = ""
    confidence = 1
    reasoning = ""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict = stripped[len("VERDICT:"):].strip()
        elif stripped.upper().startswith("CONFIDENCE:"):
            raw = stripped[len("CONFIDENCE:"):].strip()
            try:
                confidence = max(1, min(5, int(raw[0])))
            except (ValueError, IndexError):
                confidence = 1
        elif stripped.upper().startswith("REASONING:"):
            reasoning = stripped[len("REASONING:"):].strip()

    # Normalize verdict
    verdict_upper = verdict.upper()
    if "POSITION A" in verdict_upper:
        verdict = "Position A"
    elif "POSITION B" in verdict_upper:
        verdict = "Position B"
    else:
        # Fallback: scan full text
        text_upper = text.upper()
        if "POSITION A" in text_upper:
            verdict = "Position A"
        elif "POSITION B" in text_upper:
            verdict = "Position B"
        else:
            logger.warning("Could not parse verdict from response: %s", text[:200])
            verdict = "Position B"  # default to wrong when unparseable

    if not reasoning:
        reasoning = text[:500]

    return verdict, confidence, reasoning


def _is_done(response: str) -> bool:
    return response.strip().upper() == "DONE"


async def run_judgment(
    transcript: Transcript,
    world_document: str,
    judge_model: str,
    query_budget: int,
    protocol: dict,
) -> Judgment:
    seed = make_seed(
        transcript.question_id,
        transcript.transcript_index,
        judge_model,
        query_budget,
    )
    rng = random.Random(seed)
    position_a_is_correct = rng.choice([True, False])

    judge_config = protocol["judge"]
    oracle_config = protocol["oracle"]
    oracle_model: str = protocol["protocol"]["models"]["oracle"]
    judge_temperature: float = protocol["protocol"]["temperature"]["judge"]
    oracle_temperature: float = protocol["protocol"]["temperature"]["oracle"]

    position_a, position_b, debate_text = _format_transcript(transcript, position_a_is_correct)

    initial_user = judge_config["user_prompt_template"].format(
        question=transcript.question,
        position_a=position_a,
        position_b=position_b,
        debate_transcript=debate_text,
    )

    judge_messages: list[dict[str, str]] = [
        {"role": "system", "content": judge_config["system_prompt"]},
        {"role": "user", "content": initial_user},
    ]

    exchanges: list[OracleExchange] = []

    # Oracle query loop
    if query_budget > 0:
        for query_num in range(query_budget):
            remaining = query_budget - query_num
            previous = _format_previous_queries(exchanges)

            query_prompt = judge_config["query_phase_prompt"].format(
                remaining_budget=remaining,
                total_budget=query_budget,
                previous_queries=previous,
            )

            judge_messages.append({"role": "user", "content": query_prompt})

            query_response = await api.complete(
                messages=judge_messages,
                model=judge_model,
                temperature=judge_temperature,
                seed=seed + query_num,
                max_tokens=256,
            )

            judge_messages.append({"role": "assistant", "content": query_response})

            if _is_done(query_response):
                break

            # Strip "Is it supported by the text that" prefix if the judge echoes the format
            claim = query_response.strip()
            for prefix in ("Is it supported by the text that ", "is it supported by the text that "):
                if claim.startswith(prefix):
                    claim = claim[len(prefix):]
                    break

            oracle_user = oracle_config["user_prompt_template"].format(
                world_document=world_document,
                query_claim=claim,
            )

            oracle_response = await api.complete(
                messages=[
                    {"role": "system", "content": oracle_config["system_prompt"]},
                    {"role": "user", "content": oracle_user},
                ],
                model=oracle_model,
                temperature=oracle_temperature,
                seed=seed + 10000 + query_num,
                max_tokens=32,
            )

            # Normalize oracle response
            oracle_text = oracle_response.strip().upper()
            if oracle_text.startswith("YES"):
                oracle_text = "YES"
            elif oracle_text.startswith("NO"):
                oracle_text = "NO"
            else:
                oracle_text = "NOT ADDRESSED"

            exchanges.append(OracleExchange(query=claim, response=oracle_text))

            # Feed oracle result back to judge
            judge_messages.append({
                "role": "user",
                "content": f"Oracle result: {oracle_text}",
            })

    # Verdict
    query_results_str = ""
    if exchanges:
        lines = [f"Query {i}: {ex.query}\nResult: {ex.response}" for i, ex in enumerate(exchanges, 1)]
        query_results_str = "VERIFICATION RESULTS:\n\n" + "\n\n".join(lines)

    verdict_prompt = judge_config["verdict_prompt"].format(
        query_results=query_results_str,
    )

    judge_messages.append({"role": "user", "content": verdict_prompt})

    verdict_response = await api.complete(
        messages=judge_messages,
        model=judge_model,
        temperature=judge_temperature,
        seed=seed + 99999,
        max_tokens=512,
    )

    verdict, confidence, reasoning = _parse_verdict(verdict_response)
    verdict_correct = (verdict == "Position A") == position_a_is_correct

    return Judgment(
        question_id=transcript.question_id,
        transcript_index=transcript.transcript_index,
        judge_model=judge_model,
        query_budget=query_budget,
        position_a_is_correct=position_a_is_correct,
        queries_submitted=exchanges,
        queries_used=len(exchanges),
        verdict=verdict,
        verdict_correct=verdict_correct,
        confidence=confidence,
        reasoning=reasoning,
        seed=seed,
    )
