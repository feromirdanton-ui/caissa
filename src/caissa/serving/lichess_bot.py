"""lichess-bot skeleton (Act V — Teacher). SKELETON, not a working bot.

Documents the intended integration: a mentor bot for live students whose logs return to
training. No network calls are made in the build phase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BotConfig:
    token_env: str = "LICHESS_BOT_TOKEN"
    engine_name: str = "caissa-1"
    accept_challenges: bool = False


class LichessBot:
    """SKELETON. Implement the lichess-bot protocol on the cluster, not here."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def run(self) -> None:
        raise NotImplementedError(
            "LichessBot is a skeleton (Act V). See GOAL.md §4.6 and docs/TRAINING.md."
        )
