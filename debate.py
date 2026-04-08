import hashlib
import random

import api
from models import DebateTurn, Question, Transcript


def make_seed(*parts: str | int) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _format_debate_history(turns: list[DebateTurn], honest_first: bool) -> str:
    if not turns:
        return ""
    lines = ["DEBATE HISTORY:"]
    for i, turn in enumerate(turns):
        # Debater A is whoever speaks in the first slot (index 0, 2, 4, ...)
        first_slot = i % 2 == 0
        if honest_first:
            label = "Debater A" if first_slot else "Debater B"
        else:
            label = "Debater B" if first_slot else "Debater A"
        lines.append(f"Round {i + 1}:\n{label}: {turn.text}")
    return "\n\n".join(lines)


async def generate_transcript(
    question: Question,
    world_document: str,
    transcript_index: int,
    protocol: dict,
) -> Transcript:
    seed = make_seed(question.id, transcript_index)
    rng = random.Random(seed)
    honest_first = rng.choice([True, False])

    debater_model: str = protocol["protocol"]["models"]["debater"]
    temperature: float = protocol["protocol"]["temperature"]["debater"]

    honest_prompt = protocol["honest_debater"]
    dishonest_prompt = protocol["dishonest_debater"]

    turns: list[DebateTurn] = []

    for round_idx in range(protocol["protocol"]["debate_phase"]["n_rounds"]):
        for slot in range(2):
            is_honest = (slot == 0) == honest_first

            history_str = _format_debate_history(turns, honest_first)

            if is_honest:
                system = honest_prompt["system_prompt"]
                user = honest_prompt["user_prompt_template"].format(
                    world_document=world_document,
                    question=question.question,
                    correct_answer=question.correct_answer,
                    wrong_answer=question.wrong_answer,
                    debate_history=history_str,
                )
            else:
                system = dishonest_prompt["system_prompt"]
                user = dishonest_prompt["user_prompt_template"].format(
                    world_document=world_document,
                    question=question.question,
                    correct_answer=question.correct_answer,
                    wrong_answer=question.wrong_answer,
                    wrong_answer_defensibility=question.wrong_answer_defensibility,
                    debate_history=history_str,
                )

            text = await api.complete(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=debater_model,
                temperature=temperature,
                seed=seed + round_idx * 2 + slot,
                max_tokens=512,
            )

            turns.append(DebateTurn(
                speaker="honest" if is_honest else "dishonest",
                text=text,
            ))

    return Transcript(
        question_id=question.id,
        transcript_index=transcript_index,
        world=question.world,
        question=question.question,
        correct_answer=question.correct_answer,
        wrong_answer=question.wrong_answer,
        honest_first=honest_first,
        debate_transcript=turns,
        debater_model=debater_model,
        temperature=temperature,
        seed=seed,
    )
