"""Guards organ: legality mask, Queens Guard, counterfactual harness.

Article 2: the Guard is a separate module, never a method inside the model.
It imports python-chess and nothing from caissa.model.
"""

from caissa.guards.counterfactual import Verdict, counterfactual_check
from caissa.guards.legality_mask import is_legal, legal_moves
from caissa.guards.queens_guard import GuardReport, Threat, inspect_move

__all__ = [
    "Verdict",
    "counterfactual_check",
    "is_legal",
    "legal_moves",
    "GuardReport",
    "Threat",
    "inspect_move",
]
