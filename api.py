import os

from together import AsyncTogether

_client: AsyncTogether | None = None
_DRY_RUN_RESPONSE = "VERDICT: Position A\nCONFIDENCE: 1\nREASONING: Dry run — no API call made."


def _get_client() -> AsyncTogether:
    global _client
    if _client is None:
        _client = AsyncTogether()
    return _client


async def complete(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    seed: int | None = None,
    max_tokens: int = 1024,
) -> str:
    if os.environ.get("DRY_RUN"):
        return _DRY_RUN_RESPONSE

    kwargs: dict = dict(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed

    response = await _get_client().chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return content if content is not None else ""
