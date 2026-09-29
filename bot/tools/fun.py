"""Games and randomness."""
import random
import re

from bot.registry import tool


@tool("Roll dice in NdM notation, e.g. 2d6 or d20.")
def roll_dice(spec: str = "1d6") -> dict:
    m = re.fullmatch(r"(\d*)d(\d+)", spec.strip().lower())
    if not m:
        raise ValueError("use NdM, e.g. 2d6")
    n, sides = int(m.group(1) or 1), int(m.group(2))
    if not (1 <= n <= 100 and 2 <= sides <= 1000):
        raise ValueError("dice out of range")
    rolls = [random.randint(1, sides) for _ in range(n)]
    return {"rolls": rolls, "total": sum(rolls)}


@tool("Flip a coin.")
def flip_coin() -> str:
    return random.choice(["heads", "tails"])


@tool("Pick a random choice from a list of options.")
def pick_random(options: list) -> str:
    return random.choice(options)


_WORDS = ["python", "planet", "garden", "puzzle", "rocket", "island", "castle", "violin"]


@tool("Start a word-scramble game: returns a scrambled word and its length; the answer is returned "
      "under `answer` for you to keep secret from the user until they guess.")
def scramble_word() -> dict:
    w = random.choice(_WORDS)
    letters = list(w)
    random.shuffle(letters)
    return {"scrambled": "".join(letters), "length": len(w), "answer": w}
