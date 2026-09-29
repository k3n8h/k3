"""Headless training: python -m bot.train [examples.jsonl|examples.csv]"""
import json
import sys

from bot import learner, registry


def main() -> None:
    registry.load_all()
    if len(sys.argv) > 1:
        print(registry.call("train_from_file", {"path": sys.argv[1]}))
    print(json.dumps(learner.fit_all(evaluate=True), indent=2))


if __name__ == "__main__":
    main()
