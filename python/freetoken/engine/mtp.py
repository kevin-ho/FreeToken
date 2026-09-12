"""Small model-neutral MTP transaction used by opt-in adapters.

This module deliberately owns no KV pages.  The scheduler/cache lifecycle prepares the
batch and remains the owner of its resources; callers provide the normal commit/release
callbacks used by that lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class MTPProposal:
    """One-token proposal returned by a drafter after the batch is prepared."""

    logits: Any
    probabilities: Any | None
    width: int


class MTPDrafter(Protocol):
    def draft_into_batch(self, batch: Any) -> MTPProposal: ...


@dataclass(frozen=True)
class MTPResult:
    emitted_token: int
    accepted: bool
    drafted_width: int


def run_k1_transaction(
    batch: Any,
    drafter: MTPDrafter,
    verify: Callable[[Any], Any],
    commit: Callable[[Any, MTPResult], None],
    abort: Callable[[Any], None],
) -> MTPResult:
    """Run the smallest MTP transaction for one prepared request.

    ``batch`` must already have gone through the scheduler's ordinary allocation and
    preparation.  The drafter runs before target verification.  This milestone is greedy
    only: a proposal is accepted when it equals the target's first prediction; otherwise
    the target token is the bonus token.  ``commit`` is called exactly once only after
    successful verification; ``abort`` is called at most once for any exception from
    drafting, verification, or commit and must release the prepared batch idempotently.
    The callbacks retain page ownership, so this function cannot create a second cache
    owner.
    """
    try:
        proposal = drafter.draft_into_batch(batch)
        if proposal.width != 1:
            raise ValueError(f"MTP k=1 requires drafted width 1, got {proposal.width}")
        target_logits = verify(batch)
        proposed_token = _first_token(proposal.logits)
        target_token = _first_token(target_logits)
        accepted = proposed_token == target_token
        result = MTPResult(
            emitted_token=proposed_token if accepted else target_token,
            accepted=accepted,
            drafted_width=proposal.width,
        )
        commit(batch, result)
        return result
    except BaseException:
        abort(batch)
        raise


def _first_token(logits: Any) -> int:
    """Read the greedy token from a torch-like logits tensor without owning torch here."""
    values = logits.argmax(dim=-1)
    if hasattr(values, "reshape"):
        values = values.reshape(-1)[0]
    if hasattr(values, "item"):
        values = values.item()
    return int(values)


__all__ = ["MTPDrafter", "MTPProposal", "MTPResult", "run_k1_transaction"]
