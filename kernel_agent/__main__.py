"""Run the agent: ./helper run [--turns N] [--resume RUN_ID]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_agent.client import make_client, make_critic_client, MissingKeyError

from .loop import resume_run, run_agent

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    try:
        client = make_client()
    except MissingKeyError as error:
        parser.error(str(error))
    critic = make_critic_client()
    if critic is None:
        print("critic disabled: GEMINI_API_KEY is not set", flush=True)
    if args.resume:
        result = resume_run(
            root=ROOT, run_id=args.resume, client=client, turns=args.turns,
            critic=critic,
        )
    else:
        result = run_agent(root=ROOT, client=client, turns=args.turns, critic=critic)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("run_id", "status", "turn", "note")
                if result.get(k) is not None
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
