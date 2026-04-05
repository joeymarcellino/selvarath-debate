from pydantic import BaseModel


class Question(BaseModel):
    id: str
    world: str
    question: str
    correct_answer: str
    wrong_answer: str
    facts_required: list[str]
    reasoning: str
    wrong_answer_defensibility: str


class DebateTurn(BaseModel):
    speaker: str  # "honest" | "dishonest"
    text: str


class Transcript(BaseModel):
    question_id: str
    transcript_index: int
    world: str
    question: str
    correct_answer: str
    wrong_answer: str
    honest_first: bool
    debate_transcript: list[DebateTurn]
    debater_model: str
    temperature: float
    seed: int


class OracleExchange(BaseModel):
    query: str
    response: str  # "YES" | "NO" | "NOT ADDRESSED"


class Judgment(BaseModel):
    question_id: str
    transcript_index: int
    judge_model: str
    query_budget: int
    position_a_is_correct: bool
    queries_submitted: list[OracleExchange]
    queries_used: int
    verdict: str  # "Position A" | "Position B"
    verdict_correct: bool
    confidence: int  # 1-5
    reasoning: str
    seed: int
