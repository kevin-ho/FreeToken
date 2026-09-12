"""Gemma MTP adapter boundary.

A checkpoint-specific implementation is intentionally not provided until inventory facts
identify the draft computation.  The adapter is a narrow seam so the transaction does not
need to know Gemma's weight layout.
"""
from __future__ import annotations

from typing import Any, Callable

from freetoken.engine.mtp import MTPProposal


class GemmaMTPDrafter:
    """Adapt a prepared Gemma batch to the model-neutral k=1 transaction.

    ``draft_step`` is supplied by a measured checkpoint implementation.  It receives the
    already prepared batch and returns ``(logits, probabilities)``.  No Gemma tensor names,
    second page owner, or guessed draft model is constructed here.
    """

    def __init__(self, draft_step: Callable[[Any], tuple[Any, Any | None]]):
        self._draft_step = draft_step

    def draft_into_batch(self, batch: Any) -> MTPProposal:
        logits, probabilities = self._draft_step(batch)
        return MTPProposal(logits=logits, probabilities=probabilities, width=1)


__all__ = ["GemmaMTPDrafter"]
